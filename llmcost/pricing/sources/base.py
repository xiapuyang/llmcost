"""Abstract base class for all price data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from llmcost.pricing.models import ModelRecord


class PriceSource(ABC):
    """Fetch and return a list of ModelRecord from one source."""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Short identifier stored in ModelRecord.source (e.g. 'openrouter')."""

    @abstractmethod
    def fetch(self) -> list[ModelRecord]:
        """Return all available ModelRecords. Raise on unrecoverable error."""
