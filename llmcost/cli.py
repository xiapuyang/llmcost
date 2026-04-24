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
    """Run the interactive recommendation wizard."""
    import argparse as _ap

    from rich.console import Console

    from llmcost.pricing.loader import load_records
    from llmcost.recommender.display import (
        display_filter_summary,
        display_recommendations,
        render_debug_candidates,
    )
    from llmcost.recommender.engine import ModelRecommender
    from llmcost.recommender.wizard import RecommendWizard

    p = _ap.ArgumentParser(prog="llmcost recommend", add_help=False)
    p.add_argument("--debug", action="store_true", help="Show all candidates ranked by combined score")
    args, _ = p.parse_known_args(argv)

    console = Console()
    records = load_records(
        auto_refresh_days=30,
        on_refresh=lambda: console.print("[dim]Refreshing pricing data…[/dim]"),
    )
    total_count = len(records)

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
