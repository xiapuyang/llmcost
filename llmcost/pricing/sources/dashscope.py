"""Alibaba DashScope (Qwen) official pricing page scraper."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup

from llmcost.pricing.config import PROVIDERS
from llmcost.pricing.models import ModelRecord
from llmcost.pricing.sources.base import PriceSource

logger = logging.getLogger(__name__)

_PRICING_URL = PROVIDERS["dashscope"]["pricing_url"]
_PROVIDER_NAME = PROVIDERS["dashscope"]["name"]


class DashScopeSource(PriceSource):
    source_id = "dashscope"

    def fetch(self) -> list[ModelRecord]:
        try:
            resp = httpx.get(_PRICING_URL, timeout=20, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("DashScope fetch failed: %s — use overrides.yaml", e)
            return []

        fetched_at = datetime.now(timezone.utc).isoformat()
        records = self._parse(resp.text, fetched_at)
        if not records:
            logger.warning("DashScope: no models parsed — check scraper or use overrides.yaml")
        return records

    def _parse(self, html: str, fetched_at: str) -> list[ModelRecord]:
        soup = BeautifulSoup(html, "html.parser")
        records = []
        for table in soup.find_all("table"):
            for row in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) >= 3:
                    record = self._row_to_record(cells, fetched_at)
                    if record:
                        records.append(record)
        return records

    def _row_to_record(self, cells: list[str], fetched_at: str) -> ModelRecord | None:
        try:
            name = cells[0]
            inp = float(re.sub(r"[^\d.]", "", cells[1])) if cells[1] else None
            out = float(re.sub(r"[^\d.]", "", cells[2])) if cells[2] else None
            if not name or inp is None:
                return None
            model_id = name.lower().replace(" ", "-")
            return ModelRecord(
                id=f"dashscope/{model_id}",
                name=name,
                provider="dashscope",
                provider_name=_PROVIDER_NAME,
                pricing_url=_PRICING_URL,
                modality_raw="text->text",
                input_modalities=["text"],
                output_modalities=["text"],
                category="text",
                input_per_mtok=inp,
                output_per_mtok=out,
                source=self.source_id,
                fetched_at=fetched_at,
                notes="DashScope official page",
            )
        except (ValueError, IndexError):
            return None
