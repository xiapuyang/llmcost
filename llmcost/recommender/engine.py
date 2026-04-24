"""Model recommendation engine: filters records and selects tier winners."""

from __future__ import annotations

from dataclasses import dataclass

from llmcost.pricing.display.table import compute_value_ratio, compute_weighted
from llmcost.pricing.filters.arena import ArenaFilter
from llmcost.pricing.filters.blacklist import BlacklistFilter
from llmcost.pricing.models import ModelRecord
from llmcost.recommender.wizard import UserPreferences

# ── Constants ──────────────────────────────────────────────────────────────

CN_PROVIDERS: frozenset[str] = frozenset(
    {"zhipu", "minimax", "moonshotai", "dashscope", "bytedance-seed"}
)

_IMAGE_USE_CASES = {"text-to-image", "image editing"}


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    """A single model recommendation for one tier."""

    tier: str
    record: ModelRecord
    weighted_price: float
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
            winner = min(scored, key=lambda t: t[1])  # lowest weighted price
            rec = self._make_recommendation(
                "Best Value",
                winner[0],
                winner[1],
                f"Lowest weighted cost: ${winner[1]:.2f}/M tokens "
                f"(only {surviving_count} model(s) matched your criteria)",
            )
            return [rec], surviving_count

        return self._select_tiers(scored, prefs), surviving_count

    # ── Private helpers ────────────────────────────────────────────────────

    def _filter(self, prefs: UserPreferences) -> list[ModelRecord]:
        """Apply all filters and return the surviving records.

        Args:
            prefs: User preferences.

        Returns:
            Filtered list of ModelRecord objects.
        """
        records = self._blacklist_filter.apply(self._records, show_all=False)
        records = ArenaFilter(prefs.min_arena_score).apply(records)

        # Use-case → category
        target_category = "image" if prefs.use_case in _IMAGE_USE_CASES else "text"
        records = [r for r in records if r.category == target_category]

        # Vision input
        if prefs.vision_input:
            records = [r for r in records if "image" in r.input_modalities]

        # Context length
        if prefs.min_context_length is not None:
            records = [
                r for r in records
                if r.context_length is None or r.context_length >= prefs.min_context_length
            ]

        # Model source
        if prefs.model_source == "cn":
            records = [r for r in records if r.provider in CN_PROVIDERS]
        elif prefs.model_source == "us":
            records = [r for r in records if r.provider not in CN_PROVIDERS]

        # Provider subset
        if prefs.providers is not None:
            records = [r for r in records if r.provider in prefs.providers]

        # Require priceable records
        records = [
            r for r in records
            if r.input_per_mtok is not None and r.output_per_mtok is not None
        ]

        # Max weighted price
        if prefs.max_price is not None:
            records = [
                r for r in records
                if (
                    w := compute_weighted(
                        r,
                        input_ratio=prefs.input_ratio,
                        cache_hit_ratio=prefs.cache_hit_ratio,
                    )
                ) is not None and w <= prefs.max_price
            ]

        return records

    def _score(
        self, records: list[ModelRecord], prefs: UserPreferences
    ) -> list[tuple[ModelRecord, float]]:
        """Compute weighted price for each record.

        Args:
            records: Filtered ModelRecord list.
            prefs: User preferences for scoring.

        Returns:
            List of (record, weighted_price) tuples.
        """
        result = []
        for r in records:
            w = compute_weighted(
                r,
                input_ratio=prefs.input_ratio,
                cache_hit_ratio=prefs.cache_hit_ratio,
            )
            if w is not None:
                result.append((r, w))
        return result

    def _select_tiers(
        self,
        scored: list[tuple[ModelRecord, float]],
        prefs: UserPreferences,
    ) -> list[Recommendation]:
        """Select Best Value, Best Quality, and Balanced tier winners.

        Args:
            scored: List of (record, weighted_price) tuples.
            prefs: User preferences.

        Returns:
            List of up to 3 Recommendation objects (deduplicated by record id).
        """
        # Best Value: lowest weighted price
        best_value_item = min(scored, key=lambda t: t[1])
        best_value = self._make_recommendation(
            "Best Value",
            best_value_item[0],
            best_value_item[1],
            f"Lowest weighted cost: ${best_value_item[1]:.2f}/M tokens",
        )

        # Best Quality: highest Arena score (skip records with arena_score=None)
        with_arena = [(r, w) for r, w in scored if r.arena_score is not None]
        if with_arena:
            best_quality_item = max(with_arena, key=lambda t: t[0].arena_score)
            best_quality = self._make_recommendation(
                "Best Quality",
                best_quality_item[0],
                best_quality_item[1],
                f"Highest Arena score: {best_quality_item[0].arena_score}",
            )
        else:
            best_quality = None

        # Balanced: rank aggregation
        n = len(scored)
        price_sorted = sorted(range(n), key=lambda i: scored[i][1])  # lower = better
        price_rank = [0] * n
        for rank, idx in enumerate(price_sorted):
            price_rank[idx] = rank

        # Quality rank: higher arena_score = better (lower rank number)
        # Records with arena_score=None get rank n (worst)
        def quality_sort_key(i: int) -> int:
            score = scored[i][0].arena_score
            return -(score if score is not None else 0)  # negate for ascending sort

        quality_sorted = sorted(range(n), key=quality_sort_key)
        quality_rank = [0] * n
        for rank, idx in enumerate(quality_sorted):
            quality_rank[idx] = rank

        # Combined: 40% price + 60% quality, normalized to [0, 1]
        combined = [
            (0.4 * price_rank[i] / (n - 1) + 0.6 * quality_rank[i] / (n - 1), i)
            for i in range(n)
        ]
        balanced_idx = min(combined, key=lambda t: t[0])[1]
        balanced_item = scored[balanced_idx]
        balanced = self._make_recommendation(
            "Balanced",
            balanced_item[0],
            balanced_item[1],
            "Best combined cost-quality rank",
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
        rationale: str,
    ) -> Recommendation:
        """Build a Recommendation dataclass.

        Args:
            tier: Tier name string.
            record: The winning ModelRecord.
            weighted_price: Computed weighted price.
            rationale: Human-readable explanation.

        Returns:
            A Recommendation instance.
        """
        return Recommendation(
            tier=tier,
            record=record,
            weighted_price=weighted_price,
            rationale=rationale,
        )
