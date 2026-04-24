"""Static blacklist filter loaded from data/blacklist.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from llmcost.pricing.models import ModelRecord

_DEFAULT_BLACKLIST = Path(__file__).parent.parent / "data" / "blacklist.yaml"


class BlacklistFilter:
    """Filter models listed in a YAML blacklist file.

    Args:
        path: Path to the YAML blacklist file. Defaults to data/blacklist.yaml.
    """

    def __init__(self, path: Path = _DEFAULT_BLACKLIST) -> None:
        self._ids: set[str] = set()
        if path.exists():
            entries = yaml.safe_load(path.read_text()) or []
            self._ids = {e["id"] for e in entries}

    def apply(self, records: list[ModelRecord], *, show_all: bool = False) -> list[ModelRecord]:
        """Apply the blacklist to a list of ModelRecords.

        Args:
            records: List of ModelRecord instances to filter.
            show_all: If False, blacklisted models are removed entirely.
                If True, they are kept but marked with blacklisted=True.

        Returns:
            Filtered list of ModelRecord instances.
        """
        result = []
        for r in records:
            if r.id in self._ids:
                if show_all:
                    r.blacklisted = True
                    result.append(r)
            else:
                result.append(r)
        return result
