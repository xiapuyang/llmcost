# tests/test_cache.py
from pathlib import Path
from llmcost.pricing.cache import CacheManager
from llmcost.pricing.models import ModelRecord


def _make_record(**kwargs) -> ModelRecord:
    defaults = dict(
        id="test/model",
        name="Test Model",
        provider="test",
        provider_name="Test",
        pricing_url="https://example.com",
        modality_raw="text->text",
        input_modalities=["text"],
        output_modalities=["text"],
        category="text",
        input_per_mtok=1.0,
        output_per_mtok=4.0,
        source="openrouter",
        fetched_at="2026-04-23T10:00:00Z",
    )
    defaults.update(kwargs)
    return ModelRecord(**defaults)


def test_save_and_load(tmp_path):
    cache_file = tmp_path / "cache.json"
    mgr = CacheManager(cache_file)
    records = [_make_record()]
    mgr.save(records, sources={"openrouter": "2026-04-23T10:00:00Z"})
    loaded, meta = mgr.load()
    assert len(loaded) == 1
    assert loaded[0].id == "test/model"
    assert meta["sources"]["openrouter"] == "2026-04-23T10:00:00Z"


def test_load_missing_returns_empty(tmp_path):
    mgr = CacheManager(tmp_path / "nonexistent.json")
    records, meta = mgr.load()
    assert records == []
    assert meta == {}


def test_overrides_applied(tmp_path):
    overrides_file = tmp_path / "overrides.yaml"
    overrides_file.write_text(
        "- id: test/model\n"
        "  input_per_mtok: 9.99\n"
        "  notes: manual\n"
    )
    cache_file = tmp_path / "cache.json"
    mgr = CacheManager(cache_file, overrides_path=overrides_file)
    records = [_make_record()]
    result = mgr.apply_overrides(records)
    assert result[0].input_per_mtok == 9.99
    assert result[0].notes == "manual"
    assert result[0].source == "override"


def test_override_unknown_model_ignored(tmp_path):
    overrides_file = tmp_path / "overrides.yaml"
    overrides_file.write_text("- id: nonexistent/model\n  input_per_mtok: 1.0\n")
    mgr = CacheManager(tmp_path / "c.json", overrides_path=overrides_file)
    records = [_make_record()]
    result = mgr.apply_overrides(records)
    assert result[0].input_per_mtok == 1.0  # original unchanged


# ── Incremental merge (T1) ─────────────────────────────────────────────────

def test_save_preserves_records_not_in_new_fetch(tmp_path):
    """Records absent from the new fetch are preserved from the existing cache."""
    mgr = CacheManager(tmp_path / "cache.json")
    old = _make_record(id="vendor/old-model", input_per_mtok=1.0)
    new = _make_record(id="vendor/new-model", input_per_mtok=2.0)

    mgr.save([old], sources={"s": "t1"})
    mgr.save([new], sources={"s": "t2"})

    loaded, _ = mgr.load()
    ids = {r.id for r in loaded}
    assert "vendor/old-model" in ids
    assert "vendor/new-model" in ids


def test_save_overwrites_existing_record_by_id(tmp_path):
    """A record present in the new fetch overwrites its cached counterpart."""
    mgr = CacheManager(tmp_path / "cache.json")
    original = _make_record(id="vendor/model", input_per_mtok=1.0)
    updated  = _make_record(id="vendor/model", input_per_mtok=9.9)

    mgr.save([original], sources={})
    mgr.save([updated],  sources={})

    loaded, _ = mgr.load()
    assert len(loaded) == 1
    assert loaded[0].input_per_mtok == 9.9


def test_load_corrupt_json_returns_empty(tmp_path):
    """A corrupted cache.json is treated as empty instead of crashing."""
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("{invalid json{{")
    mgr = CacheManager(cache_file)
    records, meta = mgr.load()
    assert records == []
    assert meta == {}
