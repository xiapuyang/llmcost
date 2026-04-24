# llmcost pricing module

## Data sources

| Source | Providers | Method |
| --- | --- | --- |
| OpenRouter API `/api/v1/models` | All major providers (anthropic, openai, google, deepseek, x-ai, mistral, …) | HTTP JSON |
| Kimi official site | moonshotai | Playwright render + scrape |
| MiniMax official site | minimax | HTTP + BeautifulSoup, CNY→USD |
| Zhipu AI official site | zhipu | HTTP + BeautifulSoup, CNY→USD |
| DashScope official site | dashscope | HTTP + BeautifulSoup, CNY→USD |
| arena.ai leaderboards | All | HTML regex scrape (text / coding / vision / text-to-image / image-edit) |
| `data/overrides.yaml` | Manual corrections | Applied at startup, no `--refresh` needed |

Dedup rule: direct-source records take priority over OpenRouter for the same model ID.

CNY→USD rate: fetched live from frankfurter.dev at startup; fallback 0.138 on error.

---

## Filter pipeline (default)

1. **Open-source filter** — open-weights models with no Arena score are excluded (`--show-opensource` to keep)
2. **Pinned version filter** — date-pinned slugs (`-20240307`, `-0324`) always excluded; preview models with a stable counterpart excluded (`--show-pinned` to keep)
3. **Blacklist** — `data/blacklist.yaml` (`--show-all` marks them with ⚠ instead of removing)
4. **Unknown providers** — providers not in the known list and with no Arena score excluded
5. **z-ai namespace** — OpenRouter proxy namespace for Zhipu, excluded in favor of direct `zhipu/` records
6. **Arena score floor** — default 1300; models below excluded (`--min-arena-score 0` to disable)
7. **Max weighted price** — default $10/M; models above excluded (`--max-price 0` to disable)

---

## Price calculation

### Weighted price (Weighted$/M)

```
effective_input = cache_hit_ratio × cache_read_price + (1 - cache_hit_ratio) × input_price
weighted        = input_ratio × effective_input + (1 - input_ratio) × output_price
```

- `input_ratio` default **0.7** (70% input / 30% output)
- `cache_hit_ratio` default **0.5** (50% of input tokens assumed to hit prompt cache)
- Models without a `cache_read_price`: `effective_input = input_price` (no cache discount)
- Models without token pricing (image generation): `image_per_unit` is used directly

### Value ratio ($/kArena)

```
value_ratio = weighted / (arena_score / 1000)
```

Lower is better. Table is sorted by this column within each group.

---

## Arena score calculation

ELO scores are scraped from five arena.ai leaderboards and averaged with per-model-type weights:

| Model type | text | coding | vision | text-to-image | image-edit |
| --- | --- | --- | --- | --- | --- |
| Text output | 50% | 50% | — | — | — |
| Vision-in text output | 40% | 30% | 30% | — | — |
| Image output | — | — | — | 50% | 50% |

Missing categories are dropped and remaining weights renormalized.

---

## Parameters

| Flag | Default | Description |
| --- | --- | --- |
| `--input-ratio` | 0.7 | Input token share of the weighted price |
| `--cache-hit-ratio` | 0.5 | Fraction of input tokens assumed to hit prompt cache; 0 to disable |
| `--min-arena-score` | 1300 | Minimum Arena ELO score; 0 to disable |
| `--max-price` | 10.0 | Maximum weighted price in $/M; 0 to disable |
| `--output` | all | Filter by output type: `text` or `image` |
| `--provider` | all | Comma-separated slugs or aliases (claude, gemini, gpt, kimi, grok, glm, qwen, …) |
| `--vision-in` | off | Show only models that accept image input |
| `--refresh` | off | Force re-fetch all sources and overwrite cache |
| `--show-all` | off | Show blacklisted models (marked ⚠) |
| `--show-pinned` | off | Show date-pinned versions and previews superseded by a stable release |
| `--show-opensource` | off | Show open-source / open-weights models |
| `--export FILE` | off | Export results to a Markdown file |

---

## Common invocations

```bash
# Default: text models, Arena >= 1300, weighted price <= $10, 50% cache hit
python llmcost/cli.py

# No prompt cache (stateless API calls)
python llmcost/cli.py --cache-hit-ratio 0

# Output-heavy workload (e.g. document generation)
python llmcost/cli.py --input-ratio 0.3

# Filter by provider
python llmcost/cli.py --provider claude
python llmcost/cli.py --provider claude,openai

# Vision-capable models only
python llmcost/cli.py --vision-in

# Image generation models only
python llmcost/cli.py --output image

# All models, no Arena floor, no price cap
python llmcost/cli.py --min-arena-score 0 --max-price 0

# Export to Markdown
python llmcost/cli.py --export report.md

# Force refresh all data sources
python llmcost/cli.py --refresh
```
