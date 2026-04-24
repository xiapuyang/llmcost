# tests/test_filters.py
from pathlib import Path
from llmcost.pricing.models import ModelRecord
from llmcost.pricing.filters.blacklist import BlacklistFilter
from llmcost.pricing.filters.arena import ArenaFilter


def _record(model_id: str, arena_score: int | None = None) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name=model_id,
        provider=model_id.split("/")[0],
        provider_name=model_id.split("/")[0],
        pricing_url="",
        modality_raw="text->text",
        input_modalities=["text"],
        output_modalities=["text"],
        category="text",
        input_per_mtok=1.0,
        output_per_mtok=4.0,
        arena_score=arena_score,
        source="openrouter",
        fetched_at="2026-04-23T10:00:00Z",
    )


def test_blacklist_marks_model(tmp_path):
    bl_file = tmp_path / "blacklist.yaml"
    bl_file.write_text("- id: openai/old-model\n  reason: deprecated\n")
    f = BlacklistFilter(bl_file)
    records = [_record("openai/old-model"), _record("openai/gpt-4o")]
    result = f.apply(records, show_all=False)
    assert len(result) == 1
    assert result[0].id == "openai/gpt-4o"


def test_blacklist_show_all_keeps_marked(tmp_path):
    bl_file = tmp_path / "blacklist.yaml"
    bl_file.write_text("- id: openai/old-model\n  reason: deprecated\n")
    f = BlacklistFilter(bl_file)
    records = [_record("openai/old-model"), _record("openai/gpt-4o")]
    result = f.apply(records, show_all=True)
    assert len(result) == 2
    blacklisted = next(r for r in result if r.id == "openai/old-model")
    assert blacklisted.blacklisted is True
    clean = next(r for r in result if r.id == "openai/gpt-4o")
    assert clean.blacklisted is False


def test_arena_filter_excludes_below_threshold():
    f = ArenaFilter(threshold=1100)
    records = [
        _record("a/model", arena_score=1200),
        _record("b/model", arena_score=900),
        _record("c/model", arena_score=None),  # no score → keep
    ]
    result = f.apply(records)
    ids = [r.id for r in result]
    assert "a/model" in ids
    assert "c/model" in ids
    assert "b/model" not in ids
