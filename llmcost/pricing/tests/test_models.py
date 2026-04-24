"""Tests for ModelRecord dataclass and modality parsing utilities."""

from llmcost.pricing.models import ModelRecord, derive_category, parse_modality


def test_parse_modality_text_only():
    inputs, outputs = parse_modality("text->text")
    assert inputs == ["text"]
    assert outputs == ["text"]


def test_parse_modality_vision_input():
    inputs, outputs = parse_modality("text+image->text")
    assert inputs == ["text", "image"]
    assert outputs == ["text"]


def test_parse_modality_image_output():
    inputs, outputs = parse_modality("text+image->text+image")
    assert "image" in outputs


def test_derive_category_text():
    assert derive_category(["text"]) == "text"


def test_derive_category_image():
    assert derive_category(["text", "image"]) == "image"


def test_model_record_defaults():
    r = ModelRecord(
        id="anthropic/claude-sonnet-4.6",
        name="Claude Sonnet 4.6",
        provider="anthropic",
        provider_name="Anthropic",
        pricing_url="https://www.anthropic.com/pricing#api",
        modality_raw="text+image->text",
        input_modalities=["text", "image"],
        output_modalities=["text"],
        category="text",
    )
    assert r.context_length is None
    assert r.blacklisted is False
    assert r.arena_score is None


def test_model_record_to_dict_roundtrip():
    r = ModelRecord(
        id="deepseek/deepseek-chat",
        name="DeepSeek Chat",
        provider="deepseek",
        provider_name="DeepSeek",
        pricing_url="https://api-docs.deepseek.com/quick_start/pricing",
        modality_raw="text->text",
        input_modalities=["text"],
        output_modalities=["text"],
        category="text",
        input_per_mtok=0.135,
        output_per_mtok=0.55,
        context_length=64000,
        source="openrouter",
        fetched_at="2026-04-23T10:00:00Z",
    )
    d = r.to_dict()
    r2 = ModelRecord.from_dict(d)
    assert r2.id == r.id
    assert r2.input_per_mtok == r.input_per_mtok
