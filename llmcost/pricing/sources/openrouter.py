"""OpenRouter API price source."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from llmcost.pricing.config import OPENROUTER_MODELS_URL, PROVIDERS
from llmcost.pricing.models import ModelRecord, derive_category, parse_modality
from llmcost.pricing.sources.base import PriceSource

_PER_TOKEN_TO_PER_MTOK = 1_000_000

# Model ID prefixes/substrings that indicate open-source / open-weights models
_OPENSOURCE_PATTERNS = (
    "google/gemma",
    "meta-llama/",
    "microsoft/phi",
    "qwen/",
    "mistralai/mistral-",
    "mistralai/mixtral-",
    "mistralai/codestral",
    "mistralai/devstral",
    "mistralai/ministral",
    "deepseek/deepseek-r1-distill",
    "cohere/command-r",
    "01-ai/",
    "nousresearch/",
    "tngtech/",
    "arcee-ai/",
    "openai/gpt-oss-",
)


def _to_mtok(value: str | None) -> float | None:
    """Convert per-token price string to per-million-token float.

    Args:
        value: Per-token price as a string (e.g. "0.000003"), or None.

    Returns:
        Per-million-token price as a float, or None if value is falsy/zero.
    """
    if value is None:
        return None
    f = float(value)
    return round(f * _PER_TOKEN_TO_PER_MTOK, 6) if f else None


def _provider_slug(model_id: str) -> str:
    """Extract provider slug from 'provider/model-name'.

    Args:
        model_id: Model identifier in 'provider/model-name' format.

    Returns:
        The provider slug string.
    """
    return model_id.split("/")[0] if "/" in model_id else model_id


class OpenRouterSource(PriceSource):
    """Fetch model pricing records from the OpenRouter API."""

    source_id = "openrouter"

    def fetch(self) -> list[ModelRecord]:
        """Return all available ModelRecords from OpenRouter. Raise on unrecoverable error.

        Returns:
            List of ModelRecord instances, excluding free models.
        """
        resp = httpx.get(OPENROUTER_MODELS_URL, timeout=30)
        resp.raise_for_status()
        fetched_at = datetime.now(timezone.utc).isoformat()
        records = []
        for item in resp.json().get("data", []):
            record = self._parse(item, fetched_at)
            if record is not None:
                records.append(record)
        return records

    def _parse(self, item: dict, fetched_at: str) -> ModelRecord | None:
        """Parse a single model dict into a ModelRecord, or None if free.

        Args:
            item: Raw model dict from the OpenRouter API response.
            fetched_at: ISO-format UTC timestamp string for when data was fetched.

        Returns:
            A ModelRecord instance, or None if the model has no pricing.
        """
        pricing = item.get("pricing", {})
        input_per_mtok = _to_mtok(pricing.get("prompt"))
        output_per_mtok = _to_mtok(pricing.get("completion"))

        # Skip free models (no pricing) and internal routing models (negative prices)
        if not input_per_mtok and not output_per_mtok:
            return None
        if (input_per_mtok and input_per_mtok < 0) or (output_per_mtok and output_per_mtok < 0):
            return None

        modality_raw = item.get("architecture", {}).get("modality", "text->text")
        input_mods, output_mods = parse_modality(modality_raw)
        category = derive_category(output_mods)

        slug = _provider_slug(item["id"])
        provider_meta = PROVIDERS.get(slug, {})

        # image_per_unit only applies to image-output (generation) models.
        # Text models may have an "image" field for input image surcharges, which we ignore.
        image_raw = _to_mtok(pricing.get("image")) if category == "image" else None
        image_per_unit = round(image_raw / 1000, 6) if image_raw else None

        model_id = item["id"]
        opensource = any(model_id.startswith(p) for p in _OPENSOURCE_PATTERNS)
        model_slug = model_id.split("/")[-1] if "/" in model_id else model_id
        direct_id = model_slug

        supported_parameters = item.get("supported_parameters") or None
        per_request_limits = item.get("per_request_limits") or None
        description = item.get("description") or None
        created = item.get("created") or None
        canonical_slug = item.get("canonical_slug") or None
        web_search_raw = pricing.get("web_search")
        web_search_per_call = float(web_search_raw) if web_search_raw else None
        knowledge_cutoff = item.get("knowledge_cutoff") or None
        expiration_date = item.get("expiration_date") or None
        tokenizer = item.get("architecture", {}).get("tokenizer") or None

        return ModelRecord(
            id=model_id,
            name=item.get("name", model_id),
            provider=slug,
            provider_name=provider_meta.get("name", slug),
            pricing_url=provider_meta.get("pricing_url", ""),
            modality_raw=modality_raw,
            input_modalities=input_mods,
            output_modalities=output_mods,
            category=category,
            context_length=item.get("context_length"),
            max_output_tokens=item.get("top_provider", {}).get("max_completion_tokens"),
            input_per_mtok=input_per_mtok,
            output_per_mtok=output_per_mtok,
            cache_read_per_mtok=_to_mtok(pricing.get("input_cache_read")),
            cache_write_per_mtok=_to_mtok(pricing.get("input_cache_write")),
            image_per_unit=image_per_unit,
            image_unit="image" if image_per_unit else None,
            source=self.source_id,
            fetched_at=fetched_at,
            opensource=opensource,
            direct_id=direct_id,
            notes="OpenRouter rate — may differ from direct API price",
            supported_parameters=supported_parameters,
            per_request_limits=per_request_limits,
            description=description,
            created=created,
            canonical_slug=canonical_slug,
            web_search_per_call=web_search_per_call,
            knowledge_cutoff=knowledge_cutoff,
            expiration_date=expiration_date,
            tokenizer=tokenizer,
        )
