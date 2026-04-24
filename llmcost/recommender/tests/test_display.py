"""Tests for recommendation display rendering."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from llmcost.pricing.models import ModelRecord
from llmcost.recommender.display import display_recommendations
from llmcost.recommender.engine import Recommendation


def _make_record(
    record_id: str,
    provider: str = "test",
    arena_score: int | None = 1350,
    input_per_mtok: float = 1.0,
    output_per_mtok: float = 2.0,
) -> ModelRecord:
    return ModelRecord(
        id=record_id,
        provider=provider,
        provider_name=provider.capitalize(),
        name=record_id,
        pricing_url="https://example.com",
        modality_raw="text->text",
        input_per_mtok=input_per_mtok,
        output_per_mtok=output_per_mtok,
        context_length=128_000,
        input_modalities=["text"],
        output_modalities=["text"],
        category="text",
        arena_score=arena_score,
        source="test",
        fetched_at="2026-01-01T00:00:00",
    )


def _make_rec(tier: str, record_id: str, weighted_price: float = 1.0) -> Recommendation:
    return Recommendation(
        tier=tier,
        record=_make_record(record_id),
        weighted_price=weighted_price,
        rationale=f"Test rationale for {tier}",
    )


def _capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=True)
    fn(*args, console=console, **kwargs)
    return buf.getvalue()


# ── Zero survivors ────────────────────────────────────────────────────────────

def test_zero_survivors_prints_no_models_message():
    out = _capture(display_recommendations, recommendations=[], surviving_count=0)
    assert "No models matched" in out


def test_zero_survivors_suggests_relaxing_filters():
    out = _capture(display_recommendations, recommendations=[], surviving_count=0)
    assert "relax" in out.lower() or "relaxing" in out.lower()


# ── One recommendation (< 3 survivors) ────────────────────────────────────────

def test_single_recommendation_shows_panel():
    rec = _make_rec("Best Value", "p/cheap", weighted_price=0.50)
    out = _capture(display_recommendations, recommendations=[rec], surviving_count=1)
    assert "Best Value" in out
    assert "p/cheap" in out


def test_single_recommendation_shows_warning():
    rec = _make_rec("Best Value", "p/cheap")
    out = _capture(display_recommendations, recommendations=[rec], surviving_count=2)
    assert "2 model" in out
    assert "relax" in out.lower() or "option" in out.lower()


def test_single_recommendation_shows_summary_line():
    rec = _make_rec("Best Value", "p/cheap")
    out = _capture(display_recommendations, recommendations=[rec], surviving_count=2)
    assert "2 model" in out


# ── Three recommendations ──────────────────────────────────────────────────────

def test_three_recommendations_show_all_tiers():
    recs = [
        _make_rec("Best Value", "p/cheap", weighted_price=0.50),
        _make_rec("Best Quality", "p/smart", weighted_price=5.00),
        _make_rec("Balanced", "p/mid", weighted_price=2.00),
    ]
    out = _capture(display_recommendations, recommendations=recs, surviving_count=10)
    assert "Best Value" in out
    assert "Best Quality" in out
    assert "Balanced" in out


def test_three_recommendations_show_model_ids():
    recs = [
        _make_rec("Best Value", "p/cheap"),
        _make_rec("Best Quality", "p/smart"),
        _make_rec("Balanced", "p/mid"),
    ]
    out = _capture(display_recommendations, recommendations=recs, surviving_count=10)
    assert "p/cheap" in out
    assert "p/smart" in out
    assert "p/mid" in out


def test_three_recommendations_shows_summary_line():
    recs = [
        _make_rec("Best Value", "p/cheap"),
        _make_rec("Best Quality", "p/smart"),
        _make_rec("Balanced", "p/mid"),
    ]
    out = _capture(display_recommendations, recommendations=recs, surviving_count=10)
    assert "10 model" in out


def test_three_recommendations_no_warning():
    recs = [
        _make_rec("Best Value", "p/cheap"),
        _make_rec("Best Quality", "p/smart"),
        _make_rec("Balanced", "p/mid"),
    ]
    out = _capture(display_recommendations, recommendations=recs, surviving_count=10)
    # "Only N model(s)" warning should not appear when we have 3 recs
    assert "Only" not in out


# ── Panel content ─────────────────────────────────────────────────────────────

def test_panel_shows_weighted_price():
    recs = [_make_rec("Best Value", "p/cheap", weighted_price=1.23)]
    out = _capture(display_recommendations, recommendations=recs, surviving_count=1)
    assert "1.23" in out


def test_panel_shows_arena_score():
    recs = [_make_rec("Best Value", "p/model")]
    out = _capture(display_recommendations, recommendations=recs, surviving_count=1)
    assert "1350" in out


def test_panel_omits_arena_when_none():
    rec = Recommendation(
        tier="Best Value",
        record=_make_record("p/no-arena", arena_score=None),
        weighted_price=1.0,
        rationale="No arena score",
    )
    out = _capture(display_recommendations, recommendations=[rec], surviving_count=1)
    assert "Arena" not in out
