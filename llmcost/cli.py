"""Top-level CLI dispatcher: `llmcost price` and `llmcost recommend`."""

from __future__ import annotations

import argparse
import sys


def _cmd_price(argv: list[str]) -> None:
    """Run the price comparison table (existing behavior)."""
    from llmcost.pricing.cli import main as price_main

    sys.argv = [sys.argv[0]] + argv
    price_main()


def _cmd_recommend(argv: list[str]) -> None:
    """Run the recommendation wizard (interactive) or non-interactively via flags."""
    from rich.console import Console

    from llmcost.pricing.loader import load_records
    from llmcost.recommender.display import (
        display_filter_summary,
        display_recommendations,
        render_debug_candidates,
    )
    from llmcost.recommender.engine import ModelRecommender
    from llmcost.recommender.wizard import (
        RecommendWizard,
        UserPreferences,
        _USE_CASE_RATIO_DEFAULTS,
        USE_CASES,
    )

    p = argparse.ArgumentParser(prog="llmcost recommend")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--use-case", dest="use_case", metavar="NAME",
                   help=f"Non-interactive mode. One of: {', '.join(USE_CASES)}")
    p.add_argument("--vision-in", dest="vision_input", action="store_true")
    p.add_argument("--min-context-length", dest="min_context_length", type=int, metavar="N")
    p.add_argument("--model-source", dest="model_source", choices=["any", "cn", "us"], default="any")
    p.add_argument("--min-arena-score", dest="min_arena_score", type=int, default=0)
    p.add_argument("--max-price", dest="max_price", type=float, metavar="DOLLARS_PER_M")
    p.add_argument("--providers", metavar="p1,p2,...")
    p.add_argument("--require-cache-pricing", dest="require_cache_pricing", action="store_true")
    args, _ = p.parse_known_args(argv)

    console = Console()
    records = load_records(
        auto_refresh_days=30,
        on_refresh=lambda: console.print("[dim]Refreshing pricing data…[/dim]"),
    )
    total_count = len(records)

    if args.use_case is not None:
        ucd = _USE_CASE_RATIO_DEFAULTS.get(args.use_case)
        if ucd is None:
            console.print(f"[red]Unknown use case: {args.use_case!r}[/red]")
            console.print(f"[dim]Available: {', '.join(USE_CASES)}[/dim]")
            raise SystemExit(1)
        prefs = UserPreferences(
            use_case=args.use_case,
            vision_input=args.vision_input,
            input_ratio=ucd.input_token_ratio,
            input_ratio_source="use_case_def",
            cache_hit_ratio=ucd.cache_hit_ratio,
            min_context_length=args.min_context_length,
            model_source=args.model_source,
            min_arena_score=args.min_arena_score,
            max_price=args.max_price,
            providers=args.providers.split(",") if args.providers else None,
            require_cache_pricing=args.require_cache_pricing,
            required_parameters=ucd.required_parameters,
            preferred_parameters=ucd.preferred_parameters,
        )
    else:
        prefs = RecommendWizard().run()

    recommender = ModelRecommender(records)
    recommendations, surviving_count = recommender.recommend(prefs)

    console.print()
    display_filter_summary(prefs, total_count, surviving_count, console)
    console.print()
    display_recommendations(recommendations, surviving_count, console)

    if args.debug:
        console.print()
        candidates = recommender.debug_candidates(prefs)
        render_debug_candidates(candidates, prefs, console)


def main() -> None:
    """Dispatch to price or recommend subcommand."""
    p = argparse.ArgumentParser(
        prog="llmcost",
        description="LLM API price comparison and model recommendation CLI",
    )
    sub = p.add_subparsers(dest="command")
    sub.add_parser("price", help="Display the model price comparison table", add_help=False)
    sub.add_parser("recommend", help="Interactive wizard to recommend models for your use case", add_help=False)

    args, remaining = p.parse_known_args()

    if args.command == "recommend":
        _cmd_recommend(remaining)
    elif args.command == "price":
        _cmd_price(remaining)
    else:
        # Bare `llmcost` defaults to price for backwards compatibility
        _cmd_price(sys.argv[1:])


if __name__ == "__main__":
    main()
