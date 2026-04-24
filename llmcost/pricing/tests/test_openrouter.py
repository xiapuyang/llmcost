# tests/test_openrouter.py
import pytest
import respx
import httpx
from llmcost.pricing.sources.openrouter import OpenRouterSource

SAMPLE_RESPONSE = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-4.6",
            "name": "Claude Sonnet 4.6",
            "context_length": 1000000,
            "architecture": {"modality": "text+image->text"},
            "pricing": {
                "prompt": "0.000003",
                "completion": "0.000015",
                "input_cache_read": "0.0000003",
                "input_cache_write": "0.00000375",
                "image": "0.0000000075",  # vision input surcharge — must be ignored for text models
            },
        },
        {
            "id": "openai/gpt-5-image",
            "name": "GPT-5 Image",
            "context_length": 400000,
            "architecture": {"modality": "text+image->text+image"},
            "pricing": {
                "prompt": "0.00001",
                "completion": "0.00001",
                "image": "0.000167",
            },
        },
        {
            "id": "some/free-model",
            "name": "Free Model",
            "context_length": 8000,
            "architecture": {"modality": "text->text"},
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ]
}


@respx.mock
def test_fetch_returns_model_records():
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
    )
    source = OpenRouterSource()
    records = source.fetch()
    assert len(records) == 2  # free model excluded


@respx.mock
def test_text_model_fields():
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
    )
    records = {r.id: r for r in OpenRouterSource().fetch()}
    claude = records["anthropic/claude-sonnet-4.6"]
    assert claude.category == "text"
    assert claude.input_per_mtok == pytest.approx(3.0)
    assert claude.output_per_mtok == pytest.approx(15.0)
    assert claude.cache_read_per_mtok == pytest.approx(0.3)
    assert claude.input_modalities == ["text", "image"]
    assert claude.output_modalities == ["text"]
    assert claude.context_length == 1000000
    assert claude.source == "openrouter"
    # Vision input surcharge from "image" field must NOT set image_per_unit on text models
    assert claude.image_per_unit is None


@respx.mock
def test_image_model_fields():
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
    )
    records = {r.id: r for r in OpenRouterSource().fetch()}
    img = records["openai/gpt-5-image"]
    assert img.category == "image"
    assert img.image_per_unit == pytest.approx(0.167)
    assert img.image_unit == "image"


@respx.mock
def test_provider_name_from_config():
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
    )
    records = {r.id: r for r in OpenRouterSource().fetch()}
    claude = records["anthropic/claude-sonnet-4.6"]
    assert claude.provider_name == "Anthropic"
    assert claude.pricing_url == "https://www.anthropic.com/pricing#api"
