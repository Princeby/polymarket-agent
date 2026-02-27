"""
LLM Reasoning Engine
Dual-backend LLM interface (Ollama local + Groq cloud) with structured
prompt engineering for prediction market probability estimation.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import requests
from src.market import Market, format_market_for_llm

logger = logging.getLogger(__name__)

# ── Prompt Template ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert prediction market analyst and superforecaster.
Your job is to estimate the TRUE probability of prediction market events and identify when market prices are wrong.

You MUST reason systematically:
1. UNDERSTAND — What exactly is this market asking? What are the resolution criteria?
2. BASE RATE — What is the historical frequency of similar events? Start from the outside view.
3. EVIDENCE — What specific evidence do I have that adjusts the base rate?
4. ESTIMATE — Commit to a real number. Do not hedge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: THE 50-59% ZONE IS ALMOST ALWAYS WRONG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
50% means: "I have ZERO information. I cannot tell YES from NO at all."
55% means: "I have the faintest possible signal. Almost nothing."

If you've done any reasoning at all, you should be OUTSIDE this range.
Outputting 50-59% is a failure to commit, not a sign of calibration.

Audit your output before submitting:
  - Is this a genuine coin flip with no base rate? (e.g. esports game winner, zero team info) → 50% OK.
  - Do I have ANY real signal (base rates, context, common sense)? → Must go below 40% or above 60%.

Examples of CORRECT reasoning:
  Q: "Will the US militarily strike Iran on [specific date 3 days away]?"
  Wrong: 55% (hedging because uncertain)
  Right: 8% (base rate for US striking Iran on any specific day is extremely low;
              no credible intelligence suggests imminent strike; narrow 1-day window)

  Q: "Will [obscure esports team] win Game 2 vs [other team]?"
  Wrong: 55% (fake signal)
  Right: 50% (genuinely no information about these teams; coin flip is correct here)

  Q: "Will Bitcoin be between $98,000-$100,000 on [specific date]?"
  Wrong: 40% (too high for a narrow $2k band)
  Right: 8% (any specific narrow price band has low probability; the range is too tight)

  Q: "Will [politician] attend [high-profile event they are expected at]?"
  Wrong: 55% (fear of committing)
  Right: 75% (politicians almost always attend expected events; strong YES base rate)

  Q: "Will the highest temperature in [city] be exactly [narrow band] on [date]?"
  Wrong: 55% (uncertain about weather)
  Right: 15-25% (any specific narrow temperature band has low probability by base rate alone)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROBABILITY RANGES — USE THE FULL SCALE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  5-15%  : Unlikely. Low base rate AND/OR specific evidence against.
  20-35% : Below average. Some evidence against, or weak base rate.
  40-45% : Slight lean NO. You'd bet NO at fair odds, but it's close.
  50%    : GENUINE coin flip only. No information whatsoever.
  55-59% : ALMOST NEVER CORRECT. Only for the faintest possible signal.
  60-75% : Moderate YES. Clear lean based on evidence or strong base rate.
  80-95% : Strong YES. Clear evidence, dominant base rate, or near-certain.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MARKET PRICE ANCHORING RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The market price reflects crowd wisdom. Use it as your prior:
  - If market is at 60%+ and you instinctively want to say 55%: you need
    SPECIFIC evidence to deviate downward. If you have none, stay at 60-65%.
  - If market is at 15% and you instinctively want to say 55%: explain what
    specific new evidence you have. If none, stay near 15%.
  - Default: stay within ±15% of market price unless you have concrete evidence.
  - Large deviation (>20% from market) requires strong, named evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Is your estimate between 50-59%?
  → If YES: Are you outputting this because you genuinely have zero information?
    Or are you hedging due to uncertainty? Uncertainty alone is NOT a reason for 55%.
    If uncertain, use the market price ±5% instead of defaulting to 55%.
  → If NO: Proceed.

You MUST respond with ONLY valid JSON in this exact format:
{
  "reasoning": "Your step-by-step analysis (2-4 sentences). Must reference the base rate and specific evidence used.",
  "estimated_probability": 0.65,
  "confidence": "low | medium | high",
  "action": "BET_YES | BET_NO | SKIP",
  "key_factors": ["factor1", "factor2"]
}"""

USER_PROMPT_TEMPLATE = """Analyze this prediction market and estimate the true probability:

{market_data}

Remember: 50-59% is almost never correct. Commit to a real estimate using base rates and specific evidence.
Respond with ONLY valid JSON."""


# ── Analysis Result ────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Parsed result from LLM market analysis."""
    reasoning: str
    estimated_probability: float
    confidence: str        # "low", "medium", "high"
    action: str            # "BET_YES", "BET_NO", "SKIP"
    key_factors: list[str]
    raw_response: str

    @property
    def is_tradeable(self) -> bool:
        return self.action in ("BET_YES", "BET_NO")


def _parse_llm_response(raw: str) -> Optional[AnalysisResult]:
    """Parse LLM JSON response into an AnalysisResult."""
    text = raw.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning(f"No JSON found in LLM response: {raw[:200]}")
        return None

    try:
        data = json.loads(text[start:end])
        prob = float(data.get("estimated_probability", 0))
        if prob > 1:
            prob = prob / 100.0

        return AnalysisResult(
            reasoning=data.get("reasoning", ""),
            estimated_probability=max(0.0, min(1.0, prob)),
            confidence=data.get("confidence", "low").lower(),
            action=data.get("action", "SKIP").upper(),
            key_factors=data.get("key_factors", []),
            raw_response=raw,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"Failed to parse LLM response: {e}\nRaw: {raw[:300]}")
        return None


# ── LLM Backends ───────────────────────────────────────────────────────────────

class LLMBackend(ABC):
    """Abstract base for LLM inference backends."""

    @abstractmethod
    def query(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Send a prompt to the LLM and return the raw string response."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class GroqBackend(LLMBackend):
    """Groq cloud API backend (free tier)."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            )

    @property
    def name(self) -> str:
        return f"Groq ({self.model})"

    def query(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        import time as _time

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                _time.sleep(2.5)
                resp = requests.post(
                    self.base_url, headers=headers, json=payload, timeout=30
                )
                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                    _time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except requests.RequestException as e:
                if "429" not in str(e):
                    logger.error(f"Groq API error: {e}")
                return None
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected Groq response format: {e}")
                return None

        logger.error("Groq: max retries exceeded")
        return None


class OllamaBackend(LLMBackend):
    """Local Ollama backend."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    @property
    def name(self) -> str:
        return f"Ollama ({self.model})"

    def query(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_predict": 1024,
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Unexpected Ollama response format: {e}")
            return None
class CerebrasBackend(LLMBackend):
    """
    Cerebras cloud API — free tier, same Llama models as Groq but much
    higher rate limits and faster inference. OpenAI-compatible endpoint.
    Get a free key at: https://cloud.cerebras.ai
    """

    def __init__(self):
        self.api_key = os.getenv("CEREBRAS_API_KEY", "")
        self.model   = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
        self.base_url = "https://api.cerebras.ai/v1/chat/completions"

        if not self.api_key:
            raise ValueError(
                "CEREBRAS_API_KEY not set. Get a free key at https://cloud.cerebras.ai"
            )

    @property
    def name(self) -> str:
        return f"Cerebras ({self.model})"

    def query(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        import time as _time

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens":  1024,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(3):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=30)

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("retry-after", 10 * (attempt + 1)))
                    wait = max(retry_after, 10 * (attempt + 1))
                    logger.warning(f"Cerebras rate limited, waiting {wait:.0f}s (attempt {attempt+1}/3)")
                    _time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

            except requests.RequestException as e:
                logger.error(f"Cerebras API error: {e}")
                return None
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected Cerebras response format: {e}")
                return None

        logger.error("Cerebras: max retries exceeded")
        return None


class GeminiBackend(LLMBackend):
    """
    Google Gemini free tier — generous rate limits, no credit card needed.
    Uses gemini-1.5-flash by default (fast and cheap).
    Get a free key at: https://aistudio.google.com/apikey
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.base_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{{}}/generateContent?key={self.api_key}"
        )

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey"
            )

    @property
    def name(self) -> str:
        return f"Gemini ({self.model})"

    def query(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        import time as _time

        url = self.base_url.format(self.model)
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {"parts": [{"text": user_prompt}]}
            ],
            "generationConfig": {
                "temperature":     0.2,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json",
            },
        }

        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=30)

                if resp.status_code == 429:
                    wait = 15 * (attempt + 1)
                    logger.warning(f"Gemini rate limited, waiting {wait}s (attempt {attempt+1}/3)")
                    _time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

            except requests.RequestException as e:
                logger.error(f"Gemini API error: {e}")
                return None
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected Gemini response format: {e}")
                return None

        logger.error("Gemini: max retries exceeded")
        return None


# ── Public Interface ───────────────────────────────────────────────────────────

def get_backend() -> LLMBackend:
    backend_name = os.getenv("LLM_BACKEND", "groq").lower()
    if backend_name == "ollama":
        return OllamaBackend()
    elif backend_name == "cerebras":
        return CerebrasBackend()
    elif backend_name == "gemini":
        return GeminiBackend()
    elif backend_name == "groq":
        return GroqBackend()
    else:
        raise ValueError(
            f"Unknown LLM_BACKEND '{backend_name}'. "
            f"Use 'groq', 'cerebras', 'gemini', or 'ollama'."
        )


def analyze_market(
    market: Market,
    backend: LLMBackend,
    include_news: bool = True,
) -> Optional[AnalysisResult]:
    """
    Analyze a prediction market using the LLM.

    Args:
        market:       The market to analyze.
        backend:      Which LLM backend to use.
        include_news: Whether to include DuckDuckGo news context
                      (disable for backtesting).

    Returns:
        AnalysisResult if parsing succeeded, None otherwise.
    """
    market_data = format_market_for_llm(market, include_news=include_news)
    user_prompt = USER_PROMPT_TEMPLATE.format(market_data=market_data)

    logger.info(f"Analyzing market: {market.question[:60]}... [{backend.name}]")
    raw_response = backend.query(SYSTEM_PROMPT, user_prompt)

    if raw_response is None:
        logger.warning("LLM returned no response")
        return None

    result = _parse_llm_response(raw_response)
    if result is None:
        logger.warning("Failed to parse LLM response into structured format")
        return None

    logger.info(
        f"  → Prob: {result.estimated_probability:.1%} | "
        f"Confidence: {result.confidence} | Action: {result.action}"
    )
    return result


def calculate_edge(market: Market, analysis: AnalysisResult) -> dict:
    """
    Calculate the trading edge between the model's estimate and market price.

    Returns:
        Dict with raw_edge, abs_edge, direction, has_edge, threshold.
    """
    threshold = float(os.getenv("MIN_EDGE_THRESHOLD", "0.08"))
    model_prob = analysis.estimated_probability
    market_prob = market.implied_probability

    raw_edge = model_prob - market_prob

    return {
        "model_probability": model_prob,
        "market_probability": market_prob,
        "raw_edge": raw_edge,
        "abs_edge": abs(raw_edge),
        "direction": "YES" if raw_edge > 0 else "NO",
        "has_edge": abs(raw_edge) >= threshold,
        "threshold": threshold,
    }