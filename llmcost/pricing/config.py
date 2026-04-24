"""Constants and provider metadata."""

from __future__ import annotations

# ── Weighted price defaults ────────────────────────────────────────────────
DEFAULT_INPUT_RATIO = 0.7   # 70% input / 30% output; override via --input-ratio

# ── LMSYS Arena filter ─────────────────────────────────────────────────────
ARENA_DEFAULT_THRESHOLD = 1300  # models below this score are excluded by default
DEFAULT_CACHE_HIT_RATIO = 0.5   # fraction of input tokens assumed to be cache hits

# Arena score weights for text-output models.
# Missing categories are ignored and remaining weights are renormalized.
ARENA_TEXT_WEIGHTS: dict[str, float] = {"text": 0.5, "coding": 0.5}

# Arena score weights for vision-in text-output models (image in input_modalities).
ARENA_VISION_WEIGHTS: dict[str, float] = {"text": 0.4, "coding": 0.3, "vision": 0.3}

# Arena score weights for image-output models (category == "image").
ARENA_IMAGE_WEIGHTS: dict[str, float] = {"text_to_image": 0.5, "image_edit": 0.5}

# Price / context drift detection threshold (fraction, e.g. 0.20 = 20%).
PRICE_DRIFT_THRESHOLD = 0.2

# Context length comparison buffer — differences within this fraction are ignored.
CONTEXT_DRIFT_BUFFER = 0.10

# ── Currency ───────────────────────────────────────────────────────────────
FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"

# ── Provider metadata ──────────────────────────────────────────────────────
# Sourced from pydantic/genai-prices provider YAMLs on 2026-04-23.
# Self-contained: no runtime dependency on external repos.
# Update pricing_url here if a provider moves their page.
PROVIDERS: dict[str, dict] = {
    # ── International providers (via OpenRouter) ──
    "anthropic": {
        "name": "Anthropic",
        "pricing_url": "https://www.anthropic.com/pricing#api",
    },
    "openai": {
        "name": "OpenAI",
        "pricing_url": "https://platform.openai.com/docs/pricing",
    },
    "google": {
        "name": "Google",
        "pricing_url": "https://ai.google.dev/gemini-api/docs/pricing",
    },
    "deepseek": {
        "name": "DeepSeek",
        "pricing_url": "https://api-docs.deepseek.com/quick_start/pricing",
    },
    "moonshotai": {
        "name": "Moonshot AI (Kimi)",
        "pricing_url": "https://platform.moonshot.ai/docs/pricing/chat#product-pricing",
    },
    "mistral": {
        "name": "Mistral",
        "pricing_url": "https://mistral.ai/technology/#pricing",
    },
    "x-ai": {
        "name": "xAI (Grok)",
        "pricing_url": "https://x.ai/api",
    },
    # ── Chinese providers (custom scrapers) ──
    "zhipu": {
        "name": "Zhipu AI (GLM)",
        "pricing_url": "https://docs.z.ai/guides/overview/pricing",
    },
    "minimax": {
        "name": "MiniMax",
        "pricing_url": "https://platform.minimaxi.com/docs/guides/pricing-paygo",
    },
    "dashscope": {
        "name": "Alibaba Cloud (Qwen)",
        "pricing_url": "https://bailian.console.aliyun.com/pricing",
    },
    "bytedance-seed": {
        "name": "ByteDance (Seed)",
        "pricing_url": "https://www.volcengine.com/docs/82379/1544106",
    },
}

# OpenRouter API endpoint — primary real-time price source
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
