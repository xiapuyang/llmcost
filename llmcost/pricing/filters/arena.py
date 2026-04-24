"""LMSYS Chatbot Arena score filter."""

from __future__ import annotations

import logging

from llmcost.pricing.models import ModelRecord

logger = logging.getLogger(__name__)


class ArenaFilter:
    """Filter out models whose Arena score is below threshold.

    Models with no score are kept (benefit of the doubt).

    Args:
        threshold: Minimum Arena score required to keep a model.
    """

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def apply(self, records: list[ModelRecord]) -> list[ModelRecord]:
        """Apply the Arena score threshold filter.

        Args:
            records: List of ModelRecord instances to filter.

        Returns:
            Filtered list containing only models at or above the threshold,
            plus models with no Arena score.
        """
        return [
            r for r in records
            if r.arena_score is None or r.arena_score >= self.threshold
        ]
