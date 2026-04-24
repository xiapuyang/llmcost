# llmcost

A CLI tool that fetches real-time LLM API pricing from multiple sources and ranks models by cost-effectiveness against [LMSYS Chatbot Arena](https://arena.ai) ELO scores.

```
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ All Models — weighted price (input 70% / output 30%, cache 50%)                                                                     │
├──────────────────────────┬───────────────────────┬─────────┬──────────┬─────────┬─────────┬─────────────┬──────────┬───────┬───────┤
│ Model                    │ Provider              │ Input/M │ Output/M │ Context │ Max Out │ Weighted$/M │ Vision   │ Arena │ $/kA  │
├──────────────────────────┼───────────────────────┼─────────┼──────────┼─────────┼─────────┼─────────────┼──────────┼───────┼───────┤
│ ── text / vision ✗ ──   │                       │         │          │         │         │             │          │       │       │
│ deepseek-chat            │ DeepSeek              │  $0.028 │   $0.042 │   128K  │    8K   │    $0.013   │    ✗     │  1380 │ 0.009 │
│ gemini-2.5-flash         │ Google                │  $0.150 │   $0.600 │    1M   │    8K   │    $0.105   │    ✗     │  1375 │ 0.076 │
│ ...                      │                       │         │          │         │         │             │          │       │       │
╰──────────────────────────┴───────────────────────┴─────────┴──────────┴─────────┴─────────┴─────────────┴──────────┴───────┴───────╯
```

## Features

- Aggregates prices from OpenRouter (all major providers) and direct scrapers for Chinese providers (Zhipu, MiniMax, Kimi, DashScope/Qwen)
- Joins LMSYS Chatbot Arena ELO scores and sorts by **$/kArena** (weighted price per 1000 ELO points — lower is better)
- Computes **weighted price** accounting for your actual input/output ratio and prompt cache hit rate
- Detects price drift between OpenRouter and Arena's listed prices
- Manual `overrides.yaml` for corrections or providers not on OpenRouter

## Installation

Requires Python 3.11+.

```bash
pip install git+https://github.com/xiapuyang/llmcost.git
```

For Chinese provider scrapers that require JavaScript rendering, install the Playwright browser:

```bash
playwright install chromium
```

## Usage

The CLI has two subcommands:

```
llmcost price      # full comparison table with filters
llmcost recommend  # interactive wizard or non-interactive --use-case
```

### `llmcost price` — comparison table

```bash
# Default view: text models, Arena score >= 1300, weighted price <= $10/M, 50% cache hit assumed
llmcost price

# Force re-fetch all sources (OpenRouter, scrapers, Arena scores)
llmcost price --refresh

# Adjust for your workload
llmcost price --cache-hit-ratio 0         # stateless calls, no prompt cache
llmcost price --input-ratio 0.3           # output-heavy (e.g. document generation)

# Filter by provider (aliases supported)
llmcost price --provider claude
llmcost price --provider claude,openai,deepseek

# Filter by capability
llmcost price --vision-in                 # models that accept image input
llmcost price --output image              # image generation models only

# Adjust quality/price thresholds
llmcost price --min-arena-score 1350
llmcost price --max-price 5.0
llmcost price --min-arena-score 0 --max-price 0   # no filters

# Include normally-hidden models
llmcost price --show-all                  # include blacklisted models (marked ⚠)
llmcost price --show-pinned               # include date-pinned and superseded previews
llmcost price --show-opensource           # include open-weights models without Arena score

# Export
llmcost price --export report.md
```

### `llmcost recommend` — model recommendation

Interactive wizard (answers a few questions then prints Best Value / Balanced / Best Quality picks):

```bash
llmcost recommend
```

Non-interactive with `--use-case` for scripting or CI:

```bash
# Common use cases
llmcost recommend --use-case chat
llmcost recommend --use-case coding
llmcost recommend --use-case code_review
llmcost recommend --use-case rag_qa
llmcost recommend --use-case long_context_qa
llmcost recommend --use-case summarization
llmcost recommend --use-case translation
llmcost recommend --use-case structured_extraction
llmcost recommend --use-case classification
llmcost recommend --use-case chain_of_thought_reasoning
llmcost recommend --use-case math_science_solving
llmcost recommend --use-case text-to-image
llmcost recommend --use-case "image editing"
```

Combine with filters to narrow results:

```bash
# Budget coding model, US providers only, max $2/M
llmcost recommend --use-case coding --model-source us --max-price 2.0

# Vision-capable chat model with at least 64 K context
llmcost recommend --use-case chat --vision-in --min-context-length 65536

# Chinese providers only, no price cap
llmcost recommend --use-case summarization --model-source cn

# Require the provider to publish cache read pricing
llmcost recommend --use-case rag_qa --require-cache-pricing
```

Each recommendation prints three picks — **Best Value** (lowest $/kArena), **Best Quality** (highest Arena score), and **Balanced** (geometric midpoint) — along with the equivalent `llmcost price` command so you can inspect the full shortlist.

### Provider aliases

`--provider` accepts canonical slugs or these aliases:

| Alias | Resolves to |
|-------|-------------|
| `claude` | `anthropic` |
| `gemini` | `google` |
| `gpt` | `openai` |
| `grok` / `x` | `x-ai` |
| `kimi` / `moonshot` | `moonshotai` |
| `glm` | `zhipu` |
| `qwen` / `ali` / `aliyun` | `dashscope` |
| `doubao` / `seed` | `bytedance-seed` |

## Recommended configurations

Tune `--input-ratio` and `--cache-hit-ratio` to match your actual workload — the defaults (70 % input, 50 % cache) suit interactive chat but can be misleading for other patterns.

### By workload type

| Workload | Recommended flags | Why |
|----------|-------------------|-----|
| Interactive chat | _(defaults)_ | ~70 % input tokens, moderate cache reuse |
| Stateless API calls | `--cache-hit-ratio 0` | No system-prompt cache benefit |
| Document / code generation | `--input-ratio 0.3` | Output tokens dominate; penalise high output prices |
| Long-context RAG | `--cache-hit-ratio 0.8` | Large, stable context re-read on every call |
| Vision tasks | `--vision-in` | Filters to models that accept image input |
| Image generation | `--output image` | Shows text-to-image / image-edit models only |

### By quality tier

| Tier | Recommended flags | Typical $/kArena |
|------|-------------------|-----------------|
| Budget (high volume) | `--min-arena-score 1300 --max-price 2.0` | < 0.05 |
| Balanced (default) | `--min-arena-score 1300 --max-price 10.0` | 0.05 – 0.3 |
| Premium (quality-first) | `--min-arena-score 1380 --max-price 0` | any |
| Frontier (no filter) | `--min-arena-score 0 --max-price 0` | — |

### By provider focus

```bash
# Western frontier models only
llmcost price --provider claude,openai,google,deepseek,x-ai

# Chinese providers (often better value at similar quality)
llmcost price --provider zhipu,minimax,kimi,dashscope

# Free / open-weights survey (no Arena filter)
llmcost price --show-opensource --min-arena-score 0
```

### Interpreting the $/kArena column

**$/kArena** (weighted price ÷ Arena score × 1000) is the primary sort key — lower means more quality per dollar. Use it as a starting shortlist, then check:

- **Context** — a cheaper model with 8 K context is useless for long-document tasks.
- **Max output** — document generation needs a large max-output window.
- **Vision-in** — required for multimodal pipelines.
- **Provider availability** — Chinese providers may require separate API keys and have different latency profiles.

## How it works

### Data sources

| Source | Providers | Method |
|--------|-----------|--------|
| OpenRouter `/api/v1/models` | Anthropic, OpenAI, Google, DeepSeek, Mistral, xAI, … | HTTP JSON |
| Kimi official site | Moonshot AI | Playwright + scrape |
| MiniMax official site | MiniMax | HTTP + BeautifulSoup, CNY→USD |
| Zhipu AI official site | Zhipu (GLM) | HTTP + BeautifulSoup, CNY→USD |
| DashScope official site | Alibaba Cloud (Qwen) | HTTP + BeautifulSoup, CNY→USD |
| arena.ai leaderboards | All | HTML scrape (5 categories) |
| `data/overrides.yaml` | Manual corrections | Applied at every startup |

CNY→USD conversion uses a live rate from [frankfurter.dev](https://frankfurter.dev) with a fallback of 0.138.

When the same model appears in multiple sources, direct-API records take priority over OpenRouter.

### Weighted price formula

```
effective_input = cache_hit_ratio × cache_read_price + (1 − cache_hit_ratio) × input_price
weighted        = input_ratio × effective_input + (1 − input_ratio) × output_price
value_ratio     = weighted / (arena_score / 1000)     # $/kArena, lower = better
```

Defaults: `input_ratio=0.70`, `cache_hit_ratio=0.50`.

### Arena score

ELO scores are scraped from five arena.ai leaderboards and averaged with weights per model type:

| Model type | text | coding | vision | text-to-image | image-edit |
|------------|------|--------|--------|---------------|------------|
| Text output | 50% | 50% | — | — | — |
| Vision-in text | 40% | 30% | 30% | — | — |
| Image output | — | — | — | 50% | 50% |

### Manual overrides

Edit `llmcost/pricing/data/overrides.yaml` to correct prices or add providers not available on OpenRouter. Overrides are applied at every startup — no `--refresh` needed.

```yaml
- id: "deepseek/deepseek-chat"
  input_per_mtok: 0.28
  output_per_mtok: 0.42
  cache_read_per_mtok: 0.028

# create: true inserts a brand-new record
- id: "dashscope/qwen3-max"
  create: true
  name: "Qwen3-Max"
  provider: "dashscope"
  input_per_mtok: 1.2
  output_per_mtok: 6.0
```

## Contributing

1. Fork the repository and create a feature branch.
2. Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (`feat:`, `fix:`, `docs:`, `chore:`, etc.).
3. Run `pytest` before submitting a pull request.

To add a new provider source, subclass `PriceSource` in `llmcost/pricing/sources/`, register it in `fetch_all()` in `cli.py`, and add provider metadata to `PROVIDERS` in `config.py`.

## License

MIT
