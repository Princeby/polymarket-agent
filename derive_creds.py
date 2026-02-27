"""
derive_creds.py
───────────────
Run this ONCE to derive your Polymarket API credentials from your wallet
private key. No trade is placed. Copy the printed values into your .env file.

Usage:
    python derive_creds.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()

if not private_key:
    print("\n❌  POLYMARKET_PRIVATE_KEY is not set in your .env file.")
    print("    Add it and re-run:\n")
    print("    POLYMARKET_PRIVATE_KEY=your_private_key_here\n")
    exit(1)

print("\n🔑  Deriving Polymarket API credentials from your private key...")
print("    (This talks to Polymarket's CLOB server — no funds are moved)\n")

try:
    from py_clob_client.client import ClobClient

    client = ClobClient(
        host="https://clob.polymarket.com",
        key=private_key,
        chain_id=137,  # Polygon mainnet
    )

    creds = client.derive_api_key()

    print("✅  Success! Copy these three lines into your .env file:\n")
    print("─" * 55)
    print(f"POLYMARKET_API_KEY={creds.api_key}")
    print(f"POLYMARKET_API_SECRET={creds.api_secret}")
    print(f"POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")
    print("─" * 55)
    print("\n⚠️   Keep these secret — treat them like passwords.")
    print("    Never commit your .env to git.\n")

except ImportError:
    print("❌  py-clob-client is not installed. Run:\n")
    print("    pip install py-clob-client\n")
except Exception as e:
    print(f"❌  Derivation failed: {e}")
    print("\n    Common causes:")
    print("    • Private key is wrong or has extra spaces/characters")
    print("    • Your wallet hasn't interacted with Polymarket yet")
    print("      → Deposit through polymarket.com first, then retry\n")
