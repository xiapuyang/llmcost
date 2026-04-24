"""Model recommendation engine: filters records and selects tier winners."""

from __future__ import annotations

from dataclasses import dataclass

from llmcost.pricing.display.table import compute_value_ratio, compute_weighted
from llmcost.pricing.filters.blacklist import BlacklistFilter
from llmcost.pricing.filters.pipeline import RecordFilter
from llmcost.pricing.models import ModelRecord
from llmcost.recommender.wizard import UserPreferences

_IMAGE_USE_CASES = {"text-to-image", "image editing"}


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    """A single model recommendation for one tier."""

    tier: str
    record: ModelRecord
    weighted_price: float
    value_ratio: float | None
    rationale: str


# ── Engine ─────────────────────────────────────────────────────────────────

class ModelRecommender:
    """Selects best models across Best Value, Best Quality, and Balanced tiers."""

    def __init__(
        self,
        records: list[ModelRecord],
        *,
        blacklist_filter: BlacklistFilter | None = None,
    ) -> None:
        """Initialize with a list of model records.

        Args:
            records: All available ModelRecord objects.
            blacklist_filter: Optional BlacklistFilter instance for testing.
                              Defaults to BlacklistFilter() using the bundled blacklist.yaml.
        """
        self._records = records
        self._blacklist_filter = blacklist_filter or BlacklistFilter()

    def recommend(self, prefs: UserPreferences) -> tuple[list[Recommendation], int]:
        """Filter records and return recommendations plus the surviving count.

        Args:
            prefs: User preferences from the wizard.

        Returns:
            Tuple of (list[Recommendation], surviving_count).
            If fewer than 3 models survive, returns at most 1 recommendation.
        """
        surviving = self._filter(prefs)
        surviving_count = len(surviving)

        if surviving_count == 0:
            return [], 0

        scored = self._score(surviving, prefs)

        if surviving_count < 3:
            winner = min(scored, key=lambda t: t[2] if t[2] is not None else float("inf"))
            rec = self._make_recommendation(
                "Best Value",
                winner[0],
                winner[1],
                winner[2],
                f"Lowest $/kArena: {winner[2]:.3f} "
                f"(only {surviving_count} model(s) matched your criteria)",
            )
            return [rec], surviving_count

        return self._select_tiers(scored, prefs), surviving_count

    # ── Private helpers ────────────────────────────────────────────────────

    def _resolve_max_price(self, prefs: UserPreferences) -> float | None:
        """Resolve max_price_model to a weighted price ceiling.

        Looks up the SOTA model by direct_id or slug; falls back to prefs.max_price.
        """
        model_id = prefs.max_price_model
        if not model_id:
            return prefs.max_price
        for r in self._records:
            slug = r.direct_id or r.id.split("/")[-1]
            if slug == model_id:
                from llmcost.pricing.display.table import compute_weighted
                return compute_weighted(
                    r,
                    input_ratio=prefs.input_ratio,
                    cache_hit_ratio=prefs.cache_hit_ratio,
                )
        return prefs.max_price

    def _filter(self, prefs: UserPreferences) -> list[ModelRecord]:
        """Apply all filters and return the surviving records.

        Args:
            prefs: User preferences.

        Returns:
            Filtered list of ModelRecord objects.
        """
        target_category = "image" if prefs.use_case in _IMAGE_USE_CASES else "text"
        return (
            RecordFilter(self._records)
            .apply_blacklist(blacklist_filter=self._blacklist_filter)
            .exclude_redundant_pinned()
            .exclude_z_ai()
            .min_arena_score(prefs.min_arena_score)
            .category(target_category)
            .vision_input_only(enabled=prefs.vision_input)
            .min_context_length(prefs.min_context_length)
            .model_source(prefs.model_source)
            .providers_subset(prefs.providers)
            .require_pricing()
            .require_arena_score()
            .exclude_per_image_pricing()
            .has_cache_pricing(enabled=prefs.require_cache_pricing)
            .max_weighted_price(
                self._resolve_max_price(prefs),
                input_ratio=prefs.input_ratio,
                cache_hit_ratio=prefs.cache_hit_ratio,
            )
            .build()
        )

    def _score(
        self, records: list[ModelRecord], prefs: UserPreferences
    ) -> list[tuple[ModelRecord, float, float | None]]:
        """Compute weighted price and value ratio for each record.

        Args:
            records: Filtered ModelRecord list.
            prefs: User preferences for scoring.

        Returns:
            List of (record, weighted_price, value_ratio) tuples.
        """
        result = []
        for r in records:
            w = compute_weighted(
                r,
                input_ratio=prefs.input_ratio,
                cache_hit_ratio=prefs.cache_hit_ratio,
            )
            if w is None:
                continue
            vr = compute_value_ratio(
                r,
                input_ratio=prefs.input_ratio,
                cache_hit_ratio=prefs.cache_hit_ratio,
            )
            result.append((r, w, vr))
        return result

    def _select_tiers(
        self,
        scored: list[tuple[ModelRecord, float, float | None]],
        prefs: UserPreferences,
    ) -> list[Recommendation]:
        """Select Best Value, Best Quality, and Balanced tier winners.

        Args:
            scored: List of (record, weighted_price, value_ratio) tuples.
            prefs: User preferences.

        Returns:
            List of up to 3 Recommendation objects (deduplicated by record id).
        """
        # Best Value: lowest $/kArena (value_ratio)
        with_vr = [(r, w, vr) for r, w, vr in scored if vr is not None]
        best_value_item = min(with_vr or scored, key=lambda t: t[2] if t[2] is not None else float("inf"))
        best_value = self._make_recommendation(
            "Best Value",
            best_value_item[0],
            best_value_item[1],
            best_value_item[2],
            f"Lowest $/kArena: {best_value_item[2]:.3f}",
        )

        # Best Quality: highest Arena score
        with_arena = [(r, w, vr) for r, w, vr in scored if r.arena_score is not None]
        if with_arena:
            best_quality_item = max(with_arena, key=lambda t: t[0].arena_score)
            best_quality = self._make_recommendation(
                "Best Quality",
                best_quality_item[0],
                best_quality_item[1],
                best_quality_item[2],
                f"Highest Arena score: {best_quality_item[0].arena_score}",
            )
        else:
            best_quality = None

        # Balanced: rank aggregation on $/kArena and arena_score
        n = len(scored)
        vr_sorted = sorted(range(n), key=lambda i: scored[i][2] if scored[i][2] is not None else float("inf"))
        vr_rank = [0] * n
        for rank, idx in enumerate(vr_sorted):
            vr_rank[idx] = rank

        def quality_sort_key(i: int) -> int:
            score = scored[i][0].arena_score
            return -(score if score is not None else 0)

        quality_sorted = sorted(range(n), key=quality_sort_key)
        quality_rank = [0] * n
        for rank, idx in enumerate(quality_sorted):
            quality_rank[idx] = rank

        # Combined: 40% $/kArena rank + 60% quality rank, normalized to [0, 1]
        combined = [
            (0.4 * vr_rank[i] / (n - 1) + 0.6 * quality_rank[i] / (n - 1), i)
            for i in range(n)
        ]
        balanced_idx = min(combined, key=lambda t: t[0])[1]
        balanced_item = scored[balanced_idx]
        balanced = self._make_recommendation(
            "Balanced",
            balanced_item[0],
            balanced_item[1],
            balanced_item[2],
            "Best combined $/kArena + quality rank",
        )

        # Deduplicate by record id
        seen_ids: set[str] = set()
        recs: list[Recommendation] = []
        for rec in [best_value, best_quality, balanced]:
            if rec is None:
                continue
            if rec.record.id not in seen_ids:
                seen_ids.add(rec.record.id)
                recs.append(rec)
        return recs

    @staticmethod
    def _make_recommendation(
        tier: str,
        record: ModelRecord,
        weighted_price: float,
        value_ratio: float | None,
        rationale: str,
    ) -> Recommendation:
        """Build a Recommendation dataclass.

        Args:
            tier: Tier name string.
            record: The winning ModelRecord.
            weighted_price: Computed weighted price ($/M tokens).
            value_ratio: Cost per kArena ($/kArena).
            rationale: Human-readable explanation.

        Returns:
            A Recommendation instance.
        """
        return Recommendation(
            tier=tier,
            record=record,
            weighted_price=weighted_price,
            value_ratio=value_ratio,
            rationale=rationale,
        )
