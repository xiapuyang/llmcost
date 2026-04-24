"""RecordFilter: fluent builder for chaining ModelRecord filter steps."""

from __future__ import annotations

import re

from llmcost.pricing.config import PROVIDERS
from llmcost.pricing.filters.arena import ArenaFilter
from llmcost.pricing.filters.blacklist import BlacklistFilter
from llmcost.pricing.models import ModelRecord

# ── Provider sets ──────────────────────────────────────────────────────────────

CN_PROVIDERS: frozenset[str] = frozenset(
    {"zhipu", "minimax", "moonshotai", "dashscope", "bytedance-seed"}
)

# ── Pinned-version helpers ─────────────────────────────────────────────────────

_DATE_RE = re.compile(
    r"-\d{8}"              # full date: 20240307
    r"|-\d{2}-\d{2}$"     # month-day: -05-06
    r"|-\d{2}-\d{4}$"     # month-year: -09-2025
    r"|-\d{4}$"            # MMDD code: -0324, -0528, -2507
    r"|-\d{4}-\d+[km]b?$" # MMDD + context suffix: -0414-128k
)
_VERSION_RE = re.compile(r"-\d{3}$")


def _is_redundant_pinned(model_id: str, all_ids: set[str]) -> bool:
    """Return True when a model is a redundant date-pinned or preview version."""
    prefix = model_id.rsplit("/", 1)[0] + "/" if "/" in model_id else ""
    slug = model_id.split("/")[-1]

    if "preview" in slug:
        base = re.sub(r"-preview.*$", "", slug)
        # Check date in base AND in the full slug (date may appear after "-preview", e.g. "-preview-09-2025")
        if _DATE_RE.search(base) or _DATE_RE.search(slug):
            return True
        if f"{prefix}{base}" in all_ids:
            return True
        plain_preview = f"{prefix}{base}-preview"
        if slug != f"{base}-preview" and plain_preview in all_ids:
            return True
        return False

    if _DATE_RE.search(slug):
        return True

    m = _VERSION_RE.search(slug)
    if m:
        return f"{prefix}{slug[:m.start()]}" in all_ids

    return False


# ── Builder ────────────────────────────────────────────────────────────────────

class RecordFilter:
    """Fluent filter builder for lists of ModelRecord.

    Each method returns ``self`` so steps can be chained and ``.build()``
    returns the final list.

    Example::

        records = (
            RecordFilter(records)
            .apply_blacklist()
            .exclude_z_ai()
            .min_arena_score(1300)
            .category("text")
            .require_pricing()
            .build()
        )
    """

    def __init__(self, records: list[ModelRecord]) -> None:
        self._records = list(records)

    # ── Always-on noise filters ────────────────────────────────────────────────

    def exclude_opensource(self, *, enabled: bool = True) -> RecordFilter:
        """Drop open-source models that have no Arena score (no commercial API presence)."""
        if enabled:
            self._records = [
                r for r in self._records if not r.opensource or r.arena_score is not None
            ]
        return self

    def exclude_redundant_pinned(self, *, enabled: bool = True) -> RecordFilter:
        """Remove date-pinned and preview versions when a stable counterpart exists."""
        if enabled:
            all_ids = {r.id for r in self._records}
            self._records = [
                r for r in self._records if not _is_redundant_pinned(r.id, all_ids)
            ]
        return self

    def apply_blacklist(
        self,
        *,
        show_all: bool = False,
        blacklist_filter: BlacklistFilter | None = None,
    ) -> RecordFilter:
        """Apply blacklist filtering. Pass a custom BlacklistFilter for testing."""
        bf = blacklist_filter or BlacklistFilter()
        self._records = bf.apply(self._records, show_all=show_all)
        return self

    def exclude_unknown_providers(self, *, enabled: bool = True) -> RecordFilter:
        """Drop providers not in PROVIDERS config that also have no Arena score."""
        if enabled:
            self._records = [
                r for r in self._records
                if r.provider in PROVIDERS or r.arena_score is not None
            ]
        return self

    def exclude_z_ai(self) -> RecordFilter:
        """Drop z-ai/* records (OpenRouter namespace for Zhipu; direct records already present)."""
        self._records = [r for r in self._records if r.provider != "z-ai"]
        return self

    # ── Threshold / selection filters ─────────────────────────────────────────

    def exclude_per_image_pricing(self, *, enabled: bool = True) -> RecordFilter:
        """Drop records billed per image (image_per_unit set) — weighted $/M formula does not apply."""
        if enabled:
            self._records = [r for r in self._records if r.image_per_unit is None]
        return self

    def has_required_parameters(self, params: tuple[str, ...]) -> RecordFilter:
        """Exclude models that support none of *params*.

        A model passes if it has at least one of the required parameters (OR logic).
        Models with supported_parameters=None (unreported) are kept — capability unknown.
        """
        if params:
            required = set(params)
            self._records = [
                r for r in self._records
                if r.supported_parameters is None
                or bool(required.intersection(r.supported_parameters))
            ]
        return self

    def has_cache_pricing(self, enabled: bool = True) -> RecordFilter:
        """Keep only models that have cache_read_per_mtok set."""
        if enabled:
            self._records = [r for r in self._records if r.cache_read_per_mtok is not None]
        return self

    def require_arena_score(self) -> RecordFilter:
        """Drop records with no Arena score."""
        self._records = [r for r in self._records if r.arena_score is not None]
        return self

    def min_arena_score(self, threshold: int) -> RecordFilter:
        """Exclude models below *threshold* Arena score. No-op when threshold <= 0."""
        if threshold > 0:
            self._records = ArenaFilter(threshold=threshold).apply(self._records)
        return self

    def category(self, cat: str | None) -> RecordFilter:
        """Keep only records matching *cat* (e.g. 'text', 'image'). No-op when None."""
        if cat is not None:
            self._records = [r for r in self._records if r.category == cat]
        return self

    def vision_input_only(self, *, enabled: bool = True) -> RecordFilter:
        """Keep only models that accept image input."""
        if enabled:
            self._records = [r for r in self._records if "image" in r.input_modalities]
        return self

    def min_context_length(self, length: int | None) -> RecordFilter:
        """Require context_length >= *length*. No-op when None."""
        if length is not None:
            self._records = [
                r for r in self._records
                if r.context_length is None or r.context_length >= length
            ]
        return self

    def model_source(self, source: str) -> RecordFilter:
        """Filter by model origin: 'cn' (Chinese providers), 'us' (international), 'any' (no filter)."""
        if source == "cn":
            self._records = [r for r in self._records if r.provider in CN_PROVIDERS]
        elif source == "us":
            self._records = [r for r in self._records if r.provider not in CN_PROVIDERS]
        return self

    def providers_subset(self, slugs: list[str] | set[str] | None) -> RecordFilter:
        """Keep only models whose provider is in *slugs*. No-op when None."""
        if slugs is not None:
            slugs = set(slugs)
            self._records = [r for r in self._records if r.provider in slugs]
        return self

    def require_pricing(self) -> RecordFilter:
        """Drop records with null input_per_mtok or output_per_mtok."""
        self._records = [
            r for r in self._records
            if r.input_per_mtok is not None and r.output_per_mtok is not None
        ]
        return self

    def max_weighted_price(
        self,
        limit: float | None,
        *,
        input_ratio: float,
        cache_hit_ratio: float,
    ) -> RecordFilter:
        """Exclude records whose weighted price exceeds *limit* $/M. No-op when None or 0."""
        if limit:
            from llmcost.pricing.display.table import compute_weighted
            self._records = [
                r for r in self._records
                if (w := compute_weighted(r, input_ratio=input_ratio, cache_hit_ratio=cache_hit_ratio))
                is not None and w <= limit
            ]
        return self

    # ── Terminal ───────────────────────────────────────────────────────────────

    def build(self) -> list[ModelRecord]:
        """Return the filtered record list."""
        return self._records
