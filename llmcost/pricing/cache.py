"""Cache persistence and overrides loading."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from llmcost.pricing.config import PROVIDERS
from llmcost.pricing.models import ModelRecord

_DEFAULT_CACHE = Path(__file__).parent / "data" / "cache.json"
_DEFAULT_OVERRIDES = Path(__file__).parent / "data" / "overrides.yaml"

_OVERRIDE_FIELDS = (
    "input_per_mtok",
    "output_per_mtok",
    "cache_read_per_mtok",
    "cache_write_per_mtok",
    "image_per_unit",
    "image_unit",
    "context_length",
    "max_output_tokens",
    "direct_id",
    "notes",
)


class CacheManager:
    """Manages JSON cache persistence and YAML override application.

    Args:
        cache_path: Path to the JSON cache file.
        overrides_path: Path to the YAML overrides file.
    """

    def __init__(
        self,
        cache_path: Path = _DEFAULT_CACHE,
        overrides_path: Path = _DEFAULT_OVERRIDES,
    ) -> None:
        self.cache_path = cache_path
        self.overrides_path = overrides_path

    def save(self, records: list[ModelRecord], sources: dict[str, str]) -> None:
        """Merge records into the JSON cache file incrementally.

        Existing records not present in the new fetch are preserved.
        Records present in the new fetch overwrite their cached counterpart by id.

        Args:
            records: Freshly fetched ModelRecord instances.
            sources: Mapping of source name to fetch timestamp string.
        """
        existing_records, existing_meta = self.load()
        by_id = {r.id: r for r in existing_records}
        added = updated = 0
        for r in records:
            if r.id in by_id:
                updated += 1
            else:
                added += 1
            by_id[r.id] = r

        merged_sources = {**existing_meta.get("sources", {}), **sources}
        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "sources": merged_sources,
            "models": [r.to_dict() for r in by_id.values()],
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.cache_path)
        print(
            f"[cache] saved {len(by_id)} models ({added} new, {updated} updated,"
            f" {len(existing_records) - updated} preserved)",
            file=sys.stderr,
        )

    def load(self) -> tuple[list[ModelRecord], dict[str, Any]]:
        """Load records and metadata from the JSON cache file.

        Returns:
            A tuple of (records, meta) where meta excludes the 'models' key.
            Returns ([], {}) if the cache file does not exist.
        """
        if not self.cache_path.exists():
            return [], {}
        try:
            payload = json.loads(self.cache_path.read_text())
        except json.JSONDecodeError:
            print("[cache] warn: cache.json is corrupt, treating as empty", file=sys.stderr)
            return [], {}
        records = [ModelRecord.from_dict(m) for m in payload.get("models", [])]
        meta = {k: v for k, v in payload.items() if k != "models"}
        return records, meta

    def apply_overrides(self, records: list[ModelRecord]) -> list[ModelRecord]:
        """Apply YAML overrides to matching records in-place.

        Records whose id matches an override entry have the specified fields
        updated and their source set to 'override'. Unknown ids in the
        overrides file are silently ignored.

        Args:
            records: List of ModelRecord instances to patch.

        Returns:
            The same list with matching records mutated.
        """
        if not self.overrides_path.exists():
            return records
        raw = yaml.safe_load(self.overrides_path.read_text()) or []
        overrides = {entry["id"]: entry for entry in raw}
        existing_ids = {r.id for r in records}
        result = []
        for r in records:
            if r.id in overrides:
                ov = overrides[r.id]
                patched = []
                for field in _OVERRIDE_FIELDS:
                    if field in ov:
                        setattr(r, field, ov[field])
                        patched.append(field)
                r.overridden_fields = patched
                r.source = "override"
            result.append(r)
        # Create new records for override entries marked with create: true
        fetched_at = datetime.now(timezone.utc).isoformat()
        for entry in raw:
            if not entry.get("create") or entry["id"] in existing_ids:
                continue
            provider = entry.get("provider", entry["id"].split("/")[0])
            provider_meta = PROVIDERS.get(provider, {})
            r = ModelRecord(
                id=entry["id"],
                name=entry.get("name", entry["id"].split("/")[-1]),
                provider=provider,
                provider_name=provider_meta.get("name", provider),
                pricing_url=entry.get("pricing_url", provider_meta.get("pricing_url", "")),
                modality_raw=entry.get("modality_raw", "text->text"),
                input_modalities=entry.get("input_modalities", ["text"]),
                output_modalities=entry.get("output_modalities", ["text"]),
                category=entry.get("category", "text"),
                context_length=entry.get("context_length"),
                max_output_tokens=entry.get("max_output_tokens"),
                input_per_mtok=entry.get("input_per_mtok"),
                output_per_mtok=entry.get("output_per_mtok"),
                cache_read_per_mtok=entry.get("cache_read_per_mtok"),
                cache_write_per_mtok=entry.get("cache_write_per_mtok"),
                direct_id=entry.get("direct_id", entry["id"].split("/")[-1]),
                notes=entry.get("notes", ""),
                source="override",
                fetched_at=fetched_at,
            )
            result.append(r)
        return result
