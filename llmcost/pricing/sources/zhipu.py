"""Zhipu AI (GLM) official pricing page scraper."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup, Tag

from llmcost.pricing.config import PROVIDERS
from llmcost.pricing.models import ModelRecord
from llmcost.pricing.sources.base import PriceSource

logger = logging.getLogger(__name__)

_PRICING_URL = PROVIDERS["zhipu"]["pricing_url"]
_PROVIDER_NAME = PROVIDERS["zhipu"]["name"]

# Sections whose tables contain per-MTok token pricing
_TOKEN_SECTIONS = {"text models", "vision models"}


def _parse_price(raw: str) -> float | None:
    """Extract a USD per-MTok float from a cell string like '$1.4'. Returns None if free or missing."""
    if not raw or raw.strip().lower() in ("free", "-", "\\", "—"):
        return None
    m = re.search(r"\$([\d.]+)", raw)
    return float(m.group(1)) if m else None


def _model_api_id(display_name: str) -> str:
    """Convert display name (e.g. 'GLM-5.1') to API model ID (e.g. 'glm-5.1')."""
    return display_name.lower()


class ZhipuSource(PriceSource):
    """Fetch model pricing records from the Zhipu AI docs pricing page."""

    source_id = "zhipu"

    def fetch(self) -> list[ModelRecord]:
        """Return ModelRecords for all paid Zhipu text and vision models."""
        try:
            resp = httpx.get(_PRICING_URL, timeout=20, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Zhipu fetch failed: %s — use overrides.yaml to set prices", e)
            return []

        fetched_at = datetime.now(timezone.utc).isoformat()
        records = self._parse(resp.text, fetched_at)
        if not records:
            logger.warning("Zhipu: page parsed but no models found — check scraper or use overrides.yaml")
        return records

    def _parse(self, html: str, fetched_at: str) -> list[ModelRecord]:
        soup = BeautifulSoup(html, "html.parser")
        records: list[ModelRecord] = []
        current_section = ""

        for tag in soup.find_all(["h2", "h3", "table"]):
            if tag.name in ("h2", "h3"):
                current_section = tag.get_text(strip=True).replace("​", "").lower()
                continue
            if tag.name == "table" and current_section in _TOKEN_SECTIONS:
                is_vision = "vision" in current_section
                records.extend(self._parse_token_table(tag, is_vision, fetched_at))

        return records

    def _parse_token_table(self, table: Tag, is_vision: bool, fetched_at: str) -> list[ModelRecord]:
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if "model" not in headers or "input" not in headers:
            return []

        col = {name: i for i, name in enumerate(headers)}
        records = []

        for row in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if not cells:
                continue

            display_name = cells[col["model"]]
            input_price = _parse_price(cells[col["input"]])
            output_price = _parse_price(cells[col.get("output", -1)]) if "output" in col else None

            # Skip free models
            if input_price is None and output_price is None:
                continue

            cache_col = col.get("cached input")
            cache_read = _parse_price(cells[cache_col]) if cache_col is not None else None

            api_id = _model_api_id(display_name)
            model_id = f"zhipu/{api_id}"

            input_mods = ["text", "image"] if is_vision else ["text"]
            output_mods = ["text"]

            records.append(ModelRecord(
                id=model_id,
                name=display_name,
                provider="zhipu",
                provider_name=_PROVIDER_NAME,
                pricing_url=_PRICING_URL,
                modality_raw="text+image->text" if is_vision else "text->text",
                input_modalities=input_mods,
                output_modalities=output_mods,
                category="text",
                input_per_mtok=input_price,
                output_per_mtok=output_price,
                cache_read_per_mtok=cache_read,
                direct_id=api_id,
                source=self.source_id,
                fetched_at=fetched_at,
                notes="Zhipu official docs pricing",
            ))

        return records
