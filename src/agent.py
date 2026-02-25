"""
LLM Reasoning Engine
Triple-backend LLM interface (Ollama local + Groq cloud + OpenAI) with structured
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

# ── Prompt Template ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert prediction market analyst and superforecaster.
Your job is to estimate the TRUE probability of prediction market events and
identify when market prices are wrong.

You MUST reason systematically:
1. UNDERSTAND — What exactly is this market asking? What are the resolution criteria?
2. BASE RATE — What is the historical frequency of similar events? Start from the outside view.
3. EVIDENCE — What current evidence adjusts the base rate up or down?
4. ESTIMATE — Give your final probability, accounting for uncertainty.

Rules:
- Be calibrated: when you say 70%, events should happen ~70% of the time.
- Be conservative: account for unknown unknowns. Don't be overconfident.
- Be decisive: the market price already reflects conventional wisdom. 
- If your evidence points strongly in one direction, your probability 
 should reflect that — don't anchor to 50%. Underconfidence is as harmful as overconfidence.
- Consider the time horizon: how much can change before resolution?
- A market priced at $0.60 implies 60% probability. Only flag edge if your estimate
  differs from the market price by a meaningful amount.

You MUST respond with ONLY valid JSON in this exact format:
{
  "reasoning": "Your step-by-step analysis (2-4 sentences)",
  "estimated_probability": 0.65,
  "confidence": "low | medium | high",
  "action": "BET_YES | BET_NO | SKIP",
  "key_factors": ["factor1", "factor2"]
}"""

USER_PROMPT_TEMPLATE = """Analyze this prediction market and estimate the true probability:

{market_data}

Think carefully using base rates and available evidence. Respond with ONLY valid JSON."""


# ── Analysis Result ──────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Parsed result from LLM market analysis."""
    reasoning: str
    estimated_probability: float
    confidence: str  # "low", "medium", "high"
    action: str  # "BET_YES", "BET_NO", "SKIP"
    key_factors: list[str]
    raw_response: str

    @property
    def is_tradeable(self) -> bool:
        return self.action in ("BET_YES", "BET_NO")


def _parse_llm_response(raw: str) -> Optional[AnalysisResult]:
    """Parse LLM JSON response into an AnalysisResult."""
    # Try to extract JSON from the response (model might wrap it in markdown)
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning(f"No JSON found in LLM response: {raw[:200]}")
        return None

    try:
        data = json.loads(text[start:end])
        prob = float(data.get("estimated_probability", 0))
        if prob > 1:
            prob = prob / 100.0  # Handle if model outputs percentage instead of decimal

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


# ── LLM Backends ─────────────────────────────────────────────────────────────

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
    """
    Groq cloud API backend (free tier).
    Uses OpenAI-compatible chat completions endpoint.
    """

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
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
                # Rate limit: wait between calls to stay within free tier
                _time.sleep(2)
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
    """
    Local Ollama backend.
    Requires Ollama running at localhost:11434 with the model already pulled.
    """

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


class OpenAIBackend(LLMBackend):
    """
    OpenAI API backend.
    Supports all chat-capable models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, etc.

    Required env vars:
        OPENAI_API_KEY  — your OpenAI API key (https://platform.openai.com/api-keys)

    Optional env vars:
        OPENAI_MODEL    — model to use (default: gpt-4o-mini)
        OPENAI_BASE_URL — override for Azure OpenAI or custom proxy endpoints
    """

    # Models that support the json_object response format.
    # Older models (gpt-3.5-turbo < 1106, gpt-4 < 1106) don't support it.
    JSON_MODE_MODELS = {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    }

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"
        )

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Get a key at https://platform.openai.com/api-keys"
            )

    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"

    def _supports_json_mode(self) -> bool:
        """Check if the selected model supports response_format=json_object."""
        # Strip version suffixes like -2024-11-20 for matching
        base = self.model.split("-2024")[0].split("-2025")[0]
        return base in self.JSON_MODE_MODELS or self.model in self.JSON_MODE_MODELS

    def query(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        import time as _time

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # o1/o3 family uses different parameters (no system role, no temperature)
        is_reasoning_model = self.model.startswith(("o1", "o3", "o4"))

        if is_reasoning_model:
            # Reasoning models don't support system messages or temperature.
            # Merge system prompt into the user message.
            messages = [
                {
                    "role": "user",
                    "content": f"{system_prompt}\n\n{user_prompt}",
                }
            ]
            payload = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": 2048,
            }
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1024,
            }
            # Enable JSON mode if supported (guarantees parseable output)
            if self._supports_json_mode():
                payload["response_format"] = {"type": "json_object"}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    self.base_url, headers=headers, json=payload, timeout=60
                )

                if resp.status_code == 429:
                    # Respect Retry-After header if present
                    retry_after = float(resp.headers.get("retry-after", 5 * (attempt + 1)))
                    logger.warning(
                        f"OpenAI rate limited, waiting {retry_after:.0f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    _time.sleep(retry_after)
                    continue

                if resp.status_code == 400:
                    try:
                        err = resp.json().get("error", {})
                        logger.error(f"OpenAI 400 Bad Request: {err.get('message', resp.text[:200])}")
                    except Exception:
                        logger.error(f"OpenAI 400 Bad Request: {resp.text[:200]}")
                    return None  # Not retryable

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except requests.RequestException as e:
                logger.error(f"OpenAI request error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    _time.sleep(2 ** attempt)
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected OpenAI response format: {e}")
                return None

        logger.error("OpenAI: max retries exceeded")
        return None


# ── Public Interface ─────────────────────────────────────────────────────────

def get_backend() -> LLMBackend:
    """
    Factory: return the right LLM backend based on LLM_BACKEND env var.

    Supported values:
        groq    — Groq cloud API (default)
        ollama  — Local Ollama instance
        openai  — OpenAI API (gpt-4o-mini by default)
    """
    backend_name = os.getenv("LLM_BACKEND", "groq").lower()
    if backend_name == "ollama":
        return OllamaBackend()
    elif backend_name == "groq":
        return GroqBackend()
    elif backend_name == "openai":
        return OpenAIBackend()
    else:
        raise ValueError(
            f"Unknown LLM_BACKEND '{backend_name}'. Use 'groq', 'ollama', or 'openai'."
        )


def analyze_market(market: Market, backend: LLMBackend, include_news: bool = True) -> Optional[AnalysisResult]:
    """
    Analyze a prediction market using the LLM.

    Formats the market data, sends it to the model with a superforecaster prompt,
    parses the structured JSON response.

    Args:
        market: The market to analyze.
        backend: Which LLM backend to use.
        include_news: Whether to include DuckDuckGo news context (disable for backtesting).

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
        Dict with 'edge', 'direction', 'has_edge', and threshold info.
    """
    threshold = float(os.getenv("MIN_EDGE_THRESHOLD", "0.08"))
    model_prob = analysis.estimated_probability
    market_prob = market.implied_probability

    # Positive edge = model thinks YES is underpriced
    # Negative edge = model thinks NO is underpriced
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