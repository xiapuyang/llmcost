"""CLI entry point for model price comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llmcost.pricing.config import DEFAULT_INPUT_RATIO, DEFAULT_CACHE_HIT_RATIO, ARENA_DEFAULT_THRESHOLD, PRICE_DRIFT_THRESHOLD
from llmcost.pricing.display.markdown import render_markdown
from llmcost.pricing.display.table import render_drift_warnings, render_table
from llmcost.pricing.filters.pipeline import RecordFilter
from llmcost.pricing.loader import load_records
from llmcost.pricing.sources.arena_scores import detect_price_drift


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


def main() -> None:
    """Run the model price comparison CLI."""
    args = parse_args()

    def _on_refresh() -> None:
        print("No cache found — fetching…", file=sys.stderr)

    records = load_records(refresh=args.refresh, on_refresh=_on_refresh)

    records = (
        RecordFilter(records)
        .exclude_opensource(enabled=not args.show_opensource)
        .exclude_redundant_pinned(enabled=not args.show_pinned)
        .apply_blacklist(show_all=args.show_all)
        .exclude_unknown_providers(enabled=not args.show_all)
        .exclude_z_ai()
        .min_arena_score(args.min_arena_score)
        .category(args.output)
        .providers_subset(_resolve_providers(args.provider) if args.provider else None)
        .vision_input_only(enabled=args.vision_in)
        .max_weighted_price(args.max_price if args.max_price > 0 else None,
                            input_ratio=args.input_ratio,
                            cache_hit_ratio=args.cache_hit_ratio)
        .build()
    )

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
