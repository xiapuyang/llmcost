# tests/test_currency.py
import pytest
import respx
import httpx
from llmcost.pricing.currency import cny_to_usd_rate, convert_cny_to_usd

MOCK_RESPONSE = {"amount": 1.0, "base": "CNY", "date": "2026-04-23", "rates": {"USD": 0.1378}}


@respx.mock
def test_fetch_rate_success():
    respx.get("https://api.frankfurter.dev/v1/latest").mock(
        return_value=httpx.Response(200, json=MOCK_RESPONSE)
    )
    rate = cny_to_usd_rate()
    assert abs(rate - 0.1378) < 1e-6


@respx.mock
def test_fetch_rate_fallback_on_error():
    respx.get("https://api.frankfurter.dev/v1/latest").mock(
        return_value=httpx.Response(500)
    )
    rate = cny_to_usd_rate(fallback=0.14)
    assert rate == 0.14


def test_convert_cny_to_usd():
    assert abs(convert_cny_to_usd(100.0, rate=0.1378) - 13.78) < 1e-6
