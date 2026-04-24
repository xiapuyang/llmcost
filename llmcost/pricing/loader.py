"""Shared data loading pipeline: fetch, cache, deduplicate, override, arena scores."""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime, timezone

from llmcost.pricing.cache import CacheManager
from llmcost.pricing.models import ModelRecord
from llmcost.pricing.sources.arena_scores import (
    apply_arena_scores,
    fetch_arena_scores,
    load_arena_scores,
    save_arena_scores,
)
from llmcost.pricing.sources.kimi import KimiSource
from llmcost.pricing.sources.minimax import MiniMaxSource
from llmcost.pricing.sources.openrouter import OpenRouterSource
from llmcost.pricing.sources.zhipu import ZhipuSource


def _dedup(records: list[ModelRecord]) -> list[ModelRecord]:
    """Deduplicate by model ID, preferring direct-source records over OpenRouter proxies."""
    seen: dict[str, ModelRecord] = {}
    for r in records:
        if r.id not in seen:
            seen[r.id] = r
        else:
            existing = seen[r.id]
            if existing.source == "openrouter" and r.source not in ("openrouter", "override"):
                seen[r.id] = r
            elif existing.context_length is None and r.context_length is not None:
                seen[r.id] = r
    return list(seen.values())


def fetch_all() -> tuple[list[ModelRecord], dict[str, str]]:
    """Fetch from all sources and return records plus per-source fetch timestamps.

    Returns:
        Tuple of (deduplicated records, source_times dict).
    """
    sources = [OpenRouterSource(), ZhipuSource(), MiniMaxSource(), KimiSource()]
    all_records: list[ModelRecord] = []
    source_times: dict[str, str] = {}
    for src in sources:
        try:
            records = src.fetch()
            all_records.extend(records)
            if records:
                source_times[src.source_id] = records[0].fetched_at
        except Exception as e:
            print(f"[warn] {src.source_id} failed: {e}", file=sys.stderr)
    return _dedup(all_records), source_times


def load_records(
    *,
    refresh: bool = False,
    auto_refresh_days: int | None = None,
    on_refresh: Callable[[], None] | None = None,
) -> list[ModelRecord]:
    """Load records from cache (fetching when needed) with overrides and arena scores applied.

    Args:
        refresh: Force re-fetch all sources regardless of cache age.
        auto_refresh_days: Auto-refresh when the oldest fetched_at exceeds this many days.
        on_refresh: Called just before a refresh begins (use for progress messages).

    Returns:
        List of ModelRecord with overrides and arena scores applied.
    """
    cache = CacheManager()
    arena_scores = None

    if refresh:
        if on_refresh:
            on_refresh()
        records, source_times = fetch_all()
        cache.save(records, source_times)
        arena_scores, _prices = fetch_arena_scores()
        save_arena_scores(arena_scores, _prices)
    else:
        records, _meta = cache.load()
        needs_refresh = not records

        if not needs_refresh and auto_refresh_days is not None:
            fetched_ats = [r.fetched_at for r in records if r.fetched_at]
            if fetched_ats:
                try:
                    dt = datetime.fromisoformat(min(fetched_ats))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    needs_refresh = (datetime.now(timezone.utc) - dt).days > auto_refresh_days
                except ValueError:
                    pass

        if needs_refresh:
            if on_refresh:
                on_refresh()
            records, source_times = fetch_all()
            cache.save(records, source_times)
            arena_scores, _prices = fetch_arena_scores()
            save_arena_scores(arena_scores, _prices)
        else:
            arena_scores = load_arena_scores()
            if not arena_scores:
                arena_scores, _prices = fetch_arena_scores()
                save_arena_scores(arena_scores, _prices)

    records = cache.apply_overrides(records)
    if arena_scores:
        records = apply_arena_scores(records, arena_scores)
    return records
