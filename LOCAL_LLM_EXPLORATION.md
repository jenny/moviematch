# Local LLM Exploration: Qwen 2.5 14B via Ollama

Branch: `explore/local-llm-qwen`
Date: April 2026
Purpose: Comparison only — not intended for production.

---

## Motivation

Evaluate replacing the Anthropic Claude API with a locally-run open-source model to reduce costs, eliminate vendor dependency, and keep all inference on-device.

---

## Model Selection

### Key requirements for MovieMatch
- **Structured tool calling** — the app sends 3 JSON-schema tool definitions (`search_person`, `get_filmography`, `return_results`) and requires them to be called correctly every round
- **JSON output fidelity** — `return_results` must produce a valid nested array of `{title, explanation}` objects
- **Multi-turn message history** — up to 4 rounds
- Context window ≥ 4K tokens (all candidates fit in ~5K)

### Models evaluated (M1 16GB)

| Model | VRAM (Q4) | Fits M1 16GB? | Speed (tok/s) | Tool Calling | JSON Quality | Instruction Following | Context | License |
|-------|-----------|---------------|---------------|--------------|--------------|----------------------|---------|---------|
| **Qwen 2.5 7B** | ~4.5 GB | Comfortably | ~35–50 | Excellent | Very reliable | Strong | 128K | Apache 2.0 |
| **Mistral Nemo 12B** | ~7 GB | Yes | ~20–30 | Good | Strong | Good | 128K | Apache 2.0 |
| **Qwen 2.5 14B** | ~9 GB | Tight | ~15–20 | Very strong | Excellent | Very strong | 128K | Apache 2.0 |
| **Llama 3.1 8B** | ~5 GB | Comfortably | ~40–60 | Good | Good | Good | 128K | Llama Community |
| **Mistral 7B v0.3** | ~4.5 GB | Comfortably | ~40–60 | Inconsistent | Good | Moderate | 32K | Apache 2.0 |
| **Phi-3.5 Mini 3.8B** | ~2.5 GB | Comfortably | ~60–80 | Poor | Limited | Moderate | 128K | MIT |
| **Llama 3.3 70B** | ~40 GB | No | — | Near-Claude | Excellent | Excellent | 128K | Llama Community |

### Why Qwen 2.5 14B was chosen

**Qwen 2.5 7B failed validation** — after three prompt iterations it was still misfiring:
- Inferred director names from candidate film descriptions and called `search_person` without a name in the query
- Treated movie titles as person names and passed them to `search_person`
- Hallucinated filmographies from training memory instead of calling the TMDB tools

**Qwen 2.5 14B passed all validation tests** on the first attempt, including the harder case: a query naming a director with no films in the candidate list (forcing it to call `search_person` rather than shortcut).

---

## Security Mitigations

### Tracking disabled
Added to `~/.bashrc` before starting Ollama:
```bash
export DO_NOT_TRACK=1
export OLLAMA_NO_HISTORY=1
```

### Model checksum
Ollama verifies the SHA256 digest automatically during `pull`. To record the digest for version pinning:
```bash
curl -s http://localhost:11434/api/tags | python3 -m json.tool | grep digest
```
Pass it to the validation script:
```bash
python test_qwen_tools.py --model qwen2.5:14b --digest sha256:<value>
```

### Data residency
All inference runs on-device via Ollama. No query data is sent to Alibaba or any third party. Ollama exposes an OpenAI-compatible API at `localhost:11434`.

### CCP content filtering
Qwen models filter politically sensitive topics (Taiwan, Tiananmen, etc.). This has no practical impact on MovieMatch — the model only handles movie queries and tool calls.

---

## Validation

`test_qwen_tools.py` (project root, not part of the automated test suite) imports the real tool schemas and prompt builder from `claude.py` and runs two tests:

- **Test A** — genre-only query with no person name → model must call `return_results` directly
- **Test B** — query naming a director with no matching films in the candidates → model must call `search_person` first

Run with:
```bash
source venv/bin/activate && python test_qwen_tools.py --model qwen2.5:14b
```

### Prompt change made during validation
The `_build_rerank_prompt()` in `claude.py` was tightened to prevent both models from hallucinating results from training memory:
```
Only include films that appear in the candidate list above or in the filmography
results returned by get_filmography. Never include films from memory alone.
```
This change is sound for production regardless of which backend is active.

---

## Integration Design

`local_llm.py` is a drop-in replacement for `claude.py`'s `rerank()` and `rerank_stream()`. It re-uses shared logic imported directly from `claude.py`:

| Imported from `claude.py` | Purpose |
|---------------------------|---------|
| `_build_rerank_prompt` | Keeps both backends using the same prompt |
| `_extract_result_objects` | Incremental JSON parsing for streaming |
| `_filter_results` | Anti-hallucination title validation |
| `execute_tool` | TMDB tool dispatch + background filmography ingestion |
| `TOOLS` | Single source of truth for tool schemas |

Tool schemas are converted from Anthropic format to OpenAI format in `local_llm.py`.

### Switching backends

Set `USE_LOCAL_LLM=1` to activate the local backend. `main.py` selects the import at startup:

```python
if USE_LOCAL_LLM:
    from local_llm import rerank, rerank_stream
else:
    from claude import rerank, rerank_stream
```

All other config via env vars:
```
USE_LOCAL_LLM=1
LOCAL_LLM_BASE_URL=http://localhost:11434/v1   # default
LOCAL_LLM_MODEL=qwen2.5:14b                   # default
```

---

## Known Differences vs Claude API

| Behaviour | Claude API | Qwen 2.5 14B (local) |
|-----------|-----------|----------------------|
| Result count | Returns most/all relevant candidates | Returns fewer — more selective filtering |
| Model switching | Haiku (round 1) → Opus (tool rounds) | Single model throughout |
| Token cost tracking | Haiku/Opus breakdown in usage dict | `haiku_*` / `opus_*` fields zeroed; `model` key added |
| Retry on transient errors | Exponential backoff via tenacity | Single attempt; no retry logic |
| Streaming events | Anthropic `partial_json` deltas | OpenAI `delta.tool_calls` fragments |
| Inference speed | Fast (hosted) | ~15–20 tok/s on M1 16GB |
| Privacy | Query data sent to Anthropic | Fully on-device |

### On result count
The local model returns fewer results than Claude because it applies stricter relevance filtering. This is a model behaviour difference, not a bug. Possible mitigations:
1. Adjust the prompt to instruct the model to include all reasonably relevant candidates
2. Lower the filtering bar in `_filter_results()` (not recommended — risks returning weak matches)

---

## Setup

```bash
# 1. Install Ollama (tracking disabled first)
echo 'export DO_NOT_TRACK=1' >> ~/.bashrc
echo 'export OLLAMA_NO_HISTORY=1' >> ~/.bashrc
source ~/.bashrc
brew install ollama

# 2. Start Ollama and pull the model (~9 GB)
ollama serve &
ollama pull qwen2.5:14b

# 3. Validate tool calling
source venv/bin/activate
python test_qwen_tools.py --model qwen2.5:14b

# 4. Run MovieMatch with local backend
USE_LOCAL_LLM=1 uvicorn api.app:create_app --factory --reload
```

---

## Why Not Production

- **Speed**: ~15–20 tok/s on M1 is acceptable locally but not for concurrent users
- **Hardware ceiling**: 14B fits on 16GB but with memory pressure; no headroom for scale
- **No cloud hosting path**: Railway (current host) has no GPU support; GPU cloud adds cost and ops complexity that likely exceeds Anthropic API cost for low-traffic use
- **Result quality**: Tool calling is reliable but result counts and ranking quality are noticeably below Claude Opus
