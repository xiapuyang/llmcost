"""Kimi (Moonshot AI) official pricing page scraper using Playwright."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from llmcost.pricing.config import PROVIDERS
from llmcost.pricing.models import ModelRecord
from llmcost.pricing.sources.base import PriceSource

logger = logging.getLogger(__name__)

_PROVIDER_NAME = PROVIDERS["moonshotai"]["name"]
_PRICING_URL = PROVIDERS["moonshotai"]["pricing_url"]

# Pages to scrape: (url_slug, context_tokens_hint)
_PRICING_PAGES = [
    "https://platform.kimi.ai/docs/pricing/chat-k26",
    "https://platform.kimi.ai/docs/pricing/chat-k25",
    "https://platform.kimi.ai/docs/pricing/chat-k2",
]

_JS_EXTRACT = """() => {
    const tables = Array.from(document.querySelectorAll('table'));
    return tables.map(t =>
        Array.from(t.querySelectorAll('tr')).map(r =>
            Array.from(r.querySelectorAll('th,td')).map(c => c.textContent.trim())
        )
    );
}"""


def _parse_price(raw: str) -> float | None:
    """Extract USD float from a cell like '$0.60'. Returns None if not parseable."""
    m = re.search(r"\$([\d.]+)", raw)
    return float(m.group(1)) if m else None


def _parse_context(raw: str) -> int | None:
    """Extract token count from a cell like '262,144 tokens'."""
    m = re.search(r"([\d,]+)\s*tokens", raw, re.IGNORECASE)
    return int(m.group(1).replace(",", "")) if m else None


def _scrape_all_pages() -> list[list[list[str]]]:
    """Launch one Playwright browser, scrape all pricing pages, return raw table rows."""
    from playwright.sync_api import sync_playwright

    all_tables: list[list[list[str]]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for url in _PRICING_PAGES:
            try:
                page.goto(url, timeout=30_000)
                page.wait_for_selector("table", timeout=15_000)
                tables = page.evaluate(_JS_EXTRACT)
                all_tables.extend(tables)
            except Exception as e:
                logger.warning("Kimi: failed to scrape %s — %s", url, e)
        browser.close()
    return all_tables


class KimiSource(PriceSource):
    """Fetch Kimi model pricing by rendering official docs pages via Playwright."""

    source_id = "kimi"

    def fetch(self) -> list[ModelRecord]:
        """Return ModelRecords for all paid Kimi models. Raise on unrecoverable error."""
        try:
            raw_tables = _scrape_all_pages()
        except Exception as e:
            logger.warning("Kimi: Playwright scrape failed — %s", e)
            return []

        fetched_at = datetime.now(timezone.utc).isoformat()
        records: list[ModelRecord] = []
        seen_ids: set[str] = set()

        for table in raw_tables:
            if not table:
                continue
            header = [c.lower() for c in table[0]]
            if "model" not in header:
                continue
            col = {name: i for i, name in enumerate(header)}

            for row in table[1:]:
                if not row:
                    continue
                model_id = row[col["model"]].strip()
                if not model_id or model_id in seen_ids:
                    continue

                cache_hit = _parse_price(row[col.get("input price (cache hit)", -1)]) if "input price (cache hit)" in col else None
                cache_miss = _parse_price(row[col.get("input price (cache miss)", -1)]) if "input price (cache miss)" in col else None
                output = _parse_price(row[col.get("output price", -1)]) if "output price" in col else None
                context = _parse_context(row[col.get("context window", -1)]) if "context window" in col else None

                if cache_miss is None and output is None:
                    continue

                seen_ids.add(model_id)
                records.append(ModelRecord(
                    id=f"moonshotai/{model_id}",
                    name=model_id,
                    provider="moonshotai",
                    provider_name=_PROVIDER_NAME,
                    pricing_url=_PRICING_URL,
                    modality_raw="text->text",
                    input_modalities=["text"],
                    output_modalities=["text"],
                    category="text",
                    context_length=context,
                    input_per_mtok=cache_miss,
                    output_per_mtok=output,
                    cache_read_per_mtok=cache_hit,
                    direct_id=model_id,
                    source=self.source_id,
                    fetched_at=fetched_at,
                    notes="Kimi official pricing",
                ))

        if not records:
            logger.warning("Kimi: no models parsed — check scraper")
        return records
