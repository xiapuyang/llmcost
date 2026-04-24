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
    from datetime import datetime, timezone

    from rich.console import Console

    from llmcost.pricing.cache import CacheManager
    from llmcost.pricing.cli import fetch_all
    from llmcost.pricing.sources.arena_scores import (
        apply_arena_scores,
        fetch_arena_scores,
        load_arena_scores,
        save_arena_scores,
    )
    from llmcost.recommender.display import display_recommendations
    from llmcost.recommender.engine import ModelRecommender
    from llmcost.recommender.wizard import RecommendWizard

    console = Console()
    cache = CacheManager()

    records, meta = cache.load()
    needs_refresh = not records

    if not needs_refresh and records:
        fetched_ats = [r.fetched_at for r in records if r.fetched_at]
        if fetched_ats:
            oldest = min(fetched_ats)
            try:
                dt = datetime.fromisoformat(oldest)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = (datetime.now(timezone.utc) - dt).days
                needs_refresh = age_days > 30
            except ValueError:
                pass

    if needs_refresh:
        console.print("[dim]Refreshing pricing data…[/dim]")
        records, source_times = fetch_all()
        cache.save(records, source_times)
        arena_scores, arena_prices = fetch_arena_scores()
        save_arena_scores(arena_scores, arena_prices)
    else:
        arena_scores = load_arena_scores()
        if not arena_scores:
            arena_scores, arena_prices = fetch_arena_scores()
            save_arena_scores(arena_scores, arena_prices)

    records = cache.apply_overrides(records)
    records = apply_arena_scores(records, arena_scores)

    prefs = RecommendWizard().run()

    recommender = ModelRecommender(records)
    recommendations, surviving_count = recommender.recommend(prefs)

    console.print()
    display_recommendations(recommendations, surviving_count, console)


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
