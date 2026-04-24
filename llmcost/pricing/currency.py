"""CNY to USD conversion via frankfurter.app."""

from __future__ import annotations

import httpx
from llmcost.pricing.config import FRANKFURTER_URL


def cny_to_usd_rate(fallback: float = 0.138) -> float:
    """Fetch live CNY→USD rate; return fallback on any error."""
    try:
        resp = httpx.get(
            FRANKFURTER_URL,
            params={"from": "CNY", "to": "USD"},
            timeout=10,
        )
        resp.raise_for_status()
        return float(resp.json()["rates"]["USD"])
    except Exception:
        return fallback


def convert_cny_to_usd(cny: float, *, rate: float) -> float:
    """Convert a CNY price to USD using the given rate."""
    return cny * rate
