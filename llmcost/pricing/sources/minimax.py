"""MiniMax official pricing page scraper (paygo page, CNY → USD)."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup, Tag

from llmcost.pricing.config import FRANKFURTER_URL, PROVIDERS
from llmcost.pricing.models import ModelRecord
from llmcost.pricing.sources.base import PriceSource

logger = logging.getLogger(__name__)

_PRICING_URL = PROVIDERS["minimax"]["pricing_url"]
_PROVIDER_NAME = PROVIDERS["minimax"]["name"]

_CNY_TO_USD_FALLBACK = 1 / 7.25


def _fetch_cny_to_usd() -> float:
    """Fetch live CNY→USD rate from Frankfurter. Returns fallback on failure."""
    try:
        resp = httpx.get(FRANKFURTER_URL, params={"from": "CNY", "to": "USD"}, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        rate = resp.json()["rates"]["USD"]
        logger.debug("CNY→USD live rate: %s", rate)
        return float(rate)
    except Exception as e:
        logger.warning("Exchange rate fetch failed (%s), using fallback %.4f", e, _CNY_TO_USD_FALLBACK)
        return _CNY_TO_USD_FALLBACK

# Model IDs as used on the MiniMax API (lowercased)
_MODEL_ID_MAP: dict[str, str] = {
    "minimax-m2.7": "minimax-m2.7",
    "minimax-m2.7-highspeed": "minimax-m2.7-highspeed",
    "minimax-m2.5": "minimax-m2.5",
    "minimax-m2.5-highspeed": "minimax-m2.5-highspeed",
    "m2-her": "minimax-m2-her",
    "minimax-m2.1": "minimax-m2.1",
    "minimax-m2.1-highspeed": "minimax-m2.1-highspeed",
    "minimax-m2": "minimax-m2",
    "minimax-m1": "minimax-m1",
    "minimax-text-01": "minimax-text-01",
}


class MiniMaxSource(PriceSource):
    source_id = "minimax"

    def fetch(self) -> list[ModelRecord]:
        try:
            resp = httpx.get(_PRICING_URL, timeout=20, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("MiniMax fetch failed: %s — use overrides.yaml", e)
            return []

        fetched_at = datetime.now(timezone.utc).isoformat()
        cny_to_usd = _fetch_cny_to_usd()
        records = self._parse(resp.text, fetched_at, cny_to_usd)
        if not records:
            logger.warning("MiniMax: no models parsed — check scraper or use overrides.yaml")
        return records

    def _parse(self, html: str, fetched_at: str, cny_to_usd: float) -> list[ModelRecord]:
        soup = BeautifulSoup(html, "html.parser")
        records: list[ModelRecord] = []

        # Find the "文本" section heading, then collect tables until the next h2
        in_text_section = False
        for tag in soup.find_all(["h2", "table"]):
            if tag.name == "h2":
                in_text_section = tag.get_text(strip=True).replace("​", "") == "文本"
                continue
            if not in_text_section or not isinstance(tag, Tag):
                continue
            # Only parse tables whose first header column is "模型"
            headers = [th.get_text(strip=True) for th in tag.find_all("th")]
            if not headers or "模型" not in headers[0]:
                continue
            for row in tag.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                record = self._row_to_record(cells, fetched_at, cny_to_usd)
                if record:
                    records.append(record)

        return records

    def _row_to_record(self, cells: list[str], fetched_at: str, cny_to_usd: float) -> ModelRecord | None:
        if len(cells) < 3:
            return None
        try:
            name = cells[0].lstrip("*").rstrip("*").strip()
            inp_cny = float(re.sub(r"[^\d.]", "", cells[1])) if cells[1] and cells[1] != "——" else None
            out_cny = float(re.sub(r"[^\d.]", "", cells[2])) if cells[2] and cells[2] != "——" else None
            if not name or inp_cny is None:
                return None
            inp_usd = round(inp_cny * cny_to_usd, 4)
            out_usd = round(out_cny * cny_to_usd, 4) if out_cny is not None else None
            model_id = _MODEL_ID_MAP.get(name.lower(), name.lower().replace(" ", "-"))
            return ModelRecord(
                id=f"minimax/{model_id}",
                name=f"MiniMax: {name}",
                provider="minimax",
                provider_name=_PROVIDER_NAME,
                pricing_url=_PRICING_URL,
                modality_raw="text->text",
                input_modalities=["text"],
                output_modalities=["text"],
                category="text",
                input_per_mtok=inp_usd,
                output_per_mtok=out_usd,
                source=self.source_id,
                fetched_at=fetched_at,
            )
        except (ValueError, IndexError):
            return None
