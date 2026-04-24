"""CLI entry point for model price comparison."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from llmcost.pricing.cache import CacheManager
from llmcost.pricing.config import DEFAULT_INPUT_RATIO, DEFAULT_CACHE_HIT_RATIO, ARENA_DEFAULT_THRESHOLD, PRICE_DRIFT_THRESHOLD, PROVIDERS
from llmcost.pricing.display.markdown import render_markdown
from llmcost.pricing.display.table import render_drift_warnings, render_table
from llmcost.pricing.filters.arena import ArenaFilter
from llmcost.pricing.filters.blacklist import BlacklistFilter
from llmcost.pricing.sources.arena_scores import (
    apply_arena_scores,
    detect_price_drift,
    fetch_arena_scores,
    load_arena_scores,
    save_arena_scores,
)
from llmcost.pricing.sources.kimi import KimiSource
from llmcost.pricing.sources.minimax import MiniMaxSource
from llmcost.pricing.sources.openrouter import OpenRouterSource
from llmcost.pricing.sources.zhipu import ZhipuSource


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser(description="LLM API price comparison CLI")
    p.add_argument("--output", choices=["text", "image"], help="Filter by output type (text or image)")
    p.add_argument("--provider", help="Comma-separated provider slugs or aliases, e.g. claude,gemini,deepseek")
    p.add_argument(
        "--input-ratio",
        type=float,
        default=DEFAULT_INPUT_RATIO,
        help=f"Input token weight (default {DEFAULT_INPUT_RATIO}); output = 1 - this",
    )
    p.add_argument(
        "--cache-hit-ratio",
        type=float,
        default=DEFAULT_CACHE_HIT_RATIO,
        help=f"Fraction of input tokens assumed to be cache hits (default {DEFAULT_CACHE_HIT_RATIO}); 0 to disable",
    )
    p.add_argument("--vision-in", action="store_true", help="Show only models that accept image input")
    p.add_argument(
        "--min-arena-score",
        type=int,
        default=ARENA_DEFAULT_THRESHOLD,
        help=f"Exclude models below this LMSYS Arena score (default {ARENA_DEFAULT_THRESHOLD}); 0 to disable",
    )
    p.add_argument("--refresh", action="store_true", help="Force re-fetch all sources")
    p.add_argument("--export", metavar="FILE", help="Export results to Markdown file")
    p.add_argument(
        "--max-price",
        type=float,
        default=10.0,
        help="Exclude models with weighted price above this value in $/M (default 10.0); 0 to disable",
    )
    p.add_argument("--show-all", action="store_true", help="Include blacklisted models (marked ⚠)")
    p.add_argument("--show-pinned", action="store_true", help="Include date-pinned versions and preview models superseded by a stable release")
    p.add_argument("--show-opensource", action="store_true", help="Include open-source/open-weights models")
    return p.parse_args()


def fetch_all() -> tuple[list, dict]:
    """Fetch from all sources and return records plus per-source fetch timestamps.

    Returns:
        Tuple of (all_records, source_times dict).
    """
    sources = [OpenRouterSource(), ZhipuSource(), MiniMaxSource(), KimiSource()]
    all_records = []
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


_PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google",
    "claude": "anthropic",
    "gpt": "openai",
    "kimi": "moonshotai",
    "moonshot": "moonshotai",
    "x": "x-ai",
    "grok": "x-ai",
    "glm": "zhipu",
    "qwen": "dashscope",
    "ali": "dashscope",
    "aliyun": "dashscope",
    "bytedance": "bytedance-seed",
    "seed": "bytedance-seed",
    "doubao": "bytedance-seed",
}


def _resolve_providers(raw: str) -> set[str]:
    """Expand comma-separated provider input, resolving aliases to canonical slugs."""
    result = set()
    for token in (s.strip().lower() for s in raw.split(",")):
        result.add(_PROVIDER_ALIASES.get(token, token))
    return result


def _dedup(records: list) -> list:
    """Deduplicate by model ID, preferring direct-source records over OpenRouter proxies."""
    seen: dict[str, object] = {}
    for r in records:
        if r.id not in seen:
            seen[r.id] = r
        else:
            existing = seen[r.id]
            # Direct API source beats OpenRouter (which may have stale/different prices)
            if existing.source == "openrouter" and r.source not in ("openrouter", "override"):
                seen[r.id] = r
            elif existing.context_length is None and r.context_length is not None:
                seen[r.id] = r
    return list(seen.values())


# Matches date-only suffixes (non-preview pinned versions) — always redundant
_DATE_RE = re.compile(
    r"-\d{8}"               # full date: 20240307
    r"|-\d{2}-\d{2}$"       # month-day: -05-06
    r"|-\d{2}-\d{4}$"       # month-year: -09-2025
    r"|-\d{4}$"             # MMDD code: -0324, -0528, -2507
    r"|-\d{4}-\d+[km]b?$"  # MMDD + context suffix: -0414-128k, -0414-128kb
)

# Matches version suffixes like -001, -002 — only redundant when base model also exists
_VERSION_RE = re.compile(r"-\d{3}$")


def _is_redundant_pinned(model_id: str, all_ids: set[str]) -> bool:
    """Return True if the model should be filtered as a redundant pinned/preview version.

    Preview models are only filtered when a stable (non-preview) counterpart exists.
    Date-only pinned versions are always filtered.
    Version-suffix models (-001) are only filtered when a base model exists.
    """
    prefix = model_id.rsplit("/", 1)[0] + "/" if "/" in model_id else ""
    slug = model_id.split("/")[-1]

    if "preview" in slug:
        base = re.sub(r"-preview.*$", "", slug)
        # Date-stamped previews (e.g. kimi-k2-0711-preview) are always redundant
        if _DATE_RE.search(base):
            return True
        # Filter if a stable (non-preview) version exists
        if f"{prefix}{base}" in all_ids:
            return True
        # Filter variant suffixes (e.g. -customtools) if plain preview exists
        plain_preview = f"{prefix}{base}-preview"
        if slug != f"{base}-preview" and plain_preview in all_ids:
            return True
        return False

    if _DATE_RE.search(slug):
        return True

    # Version suffix (-001 etc.): only redundant when base model also exists
    m = _VERSION_RE.search(slug)
    if m:
        base_slug = slug[: m.start()]
        return f"{prefix}{base_slug}" in all_ids

    return False


def main() -> None:
    """Run the model price comparison CLI."""
    args = parse_args()
    cache = CacheManager()

    if args.refresh:
        records, source_times = fetch_all()
        cache.save(records, source_times)
        arena_scores, arena_prices = fetch_arena_scores()
        save_arena_scores(arena_scores, arena_prices)
    else:
        records, meta = cache.load()
        if not records:
            print("No cache found — fetching…", file=sys.stderr)
            records, source_times = fetch_all()
            cache.save(records, source_times)
        arena_scores = load_arena_scores()
        if not arena_scores:
            arena_scores, arena_prices = fetch_arena_scores()
            save_arena_scores(arena_scores, arena_prices)

    # Always apply overrides after loading so changes take effect without --refresh
    records = cache.apply_overrides(records)

    records = apply_arena_scores(records, arena_scores)

    # Filter: open-source/open-weights models without commercial deployment.
    # Keep open-source models that have an Arena score (they have meaningful commercial API presence).
    if not args.show_opensource:
        records = [r for r in records if not r.opensource or r.arena_score is not None]

    # Filter: redundant preview and date-pinned versions (default on, disable with --show-pinned)
    if not args.show_pinned:
        all_ids = {r.id for r in records}
        records = [r for r in records if not _is_redundant_pinned(r.id, all_ids)]

    # Filter: blacklist
    records = BlacklistFilter().apply(records, show_all=args.show_all)

    # Filter: unknown providers with no Arena score (obscure OpenRouter-hosted models)
    if not args.show_all:
        records = [r for r in records if r.provider in PROVIDERS or r.arena_score is not None]

    # Filter: shadowed providers covered by direct scrapers (z-ai = OpenRouter namespace for Zhipu AI)
    records = [r for r in records if r.provider != "z-ai"]

    # Filter: arena score (default threshold applied unless explicitly set to 0)
    if args.min_arena_score > 0:
        records = ArenaFilter(threshold=args.min_arena_score).apply(records)

    # Filter: output type
    if args.output:
        records = [r for r in records if r.category == args.output]

    # Filter: provider
    if args.provider:
        slugs = _resolve_providers(args.provider)
        records = [r for r in records if r.provider in slugs]

    # Filter: vision input
    if args.vision_in:
        records = [r for r in records if "image" in r.input_modalities]

    # Filter: max weighted price
    if args.max_price > 0:
        from llmcost.pricing.display.table import compute_weighted
        records = [
            r for r in records
            if (w := compute_weighted(r, input_ratio=args.input_ratio, cache_hit_ratio=args.cache_hit_ratio)) is None
            or w <= args.max_price
        ]

    # Output
    if args.export:
        md = render_markdown(records, input_ratio=args.input_ratio, cache_hit_ratio=args.cache_hit_ratio)
        Path(args.export).write_text(md)
        print(f"Exported to {args.export}")
    else:
        render_table(records, input_ratio=args.input_ratio, cache_hit_ratio=args.cache_hit_ratio, category=args.output)
        drifted = detect_price_drift(records, threshold=PRICE_DRIFT_THRESHOLD)
        render_drift_warnings(drifted)


if __name__ == "__main__":
    main()
