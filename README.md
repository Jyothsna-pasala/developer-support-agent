# Developer Support Agent: Local LLM Agent with Integrated Tools

A modular Python AI agent that helps developers with Python, React, TypeScript,
REST API integration, troubleshooting and documentation. It's a learning
project for the Microsoft Applied Skills credential **Develop an agent with
integrated tools**.

It runs in three modes:

| Mode | Who decides which tool to call | Needs Azure? | Needs internet? | Cost |
|---|---|---|---|---|
| `ollama` (default) | A **local LLM** running in [Ollama](https://ollama.com) | No | Only to download the model once | Free |
| `local` | Deterministic Python rules (no LLM) | No | No | Free |
| `foundry` | A model hosted in Microsoft Foundry | Yes | Yes | Pay-as-you-go |

> **Honesty note:** `ollama` mode is a genuine LLM agent, but it runs on your
> machine. It is **not** Microsoft Foundry. It teaches the same agent loop
> (instructions, tool schemas, model-chosen tool calls, tool results,
> conversation state) that Foundry uses. `app/foundry_agent.py` is written
> against the Foundry SDK but has not been run against a live Foundry project.

---

## 1. Architecture

```
                        ┌──────────────────────────────┐
 python -m app.agent ──▶│ CLI (run_cli)                │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │ DeveloperSupportAgent        │  name, system instructions,
                        │ input validation, history,   │  ConversationHistory (session
                        │ error handling, logging      │  transcript), logging
                        └──────────────┬───────────────┘
                                       │ AgentBackend protocol
     ┌──────────────────────────┬──────┴─────────────────┬───────────────────────────┐
     ▼                          ▼                        ▼                           │
┌──────────────────┐   ┌────────────────────┐   ┌───────────────────────┐            │
│ FallbackBackend  │──▶│ OllamaAgentBackend │   │ FoundryAgentBackend   │            │
│ (ollama mode)    │   │ LLM picks tools,   │   │ (foundry mode only,   │            │
│ on error ───────┐│   │ tool-calling loop  │   │  Azure SDK)           │            │
└─────────────────┼┘   └─────────┬──────────┘   └───────────┬───────────┘            │
                  │              │ HTTP (requests)          │                        │
                  │              ▼                          │                        │
                  │      Ollama server :11434               │                        │
                  ▼                                         │                        │
        ┌───────────────────────┐                           │                        │
        │ LocalSimulatedBackend │◀── local mode ────────────┼────────────────────────┘
        │ rule-based routing    │                           │
        └──────────┬────────────┘                           │
                   └──────────────┬─────────────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │ tools.py: ToolSpec registry,    │──▶ data/knowledge_base.txt
                 │ JSON schemas, execute_tool()    │
                 └─────────────────────────────────┘
          config.py: .env loading and validation (used by everything above)
```

### Project structure

```
foundry-agent-project/
├── app/
│   ├── __init__.py
│   ├── agent.py            Agent definition, rule-based backend, FallbackBackend, CLI
│   ├── ollama_agent.py     Ollama HTTP client + LLM tool-calling loop     (new)
│   ├── tools.py            3 tools, JSON schemas, validation, executor
│   ├── config.py           .env loading and validation (now includes Ollama)
│   ├── conversation.py     Shared types: history, replies, backend protocol
│   └── foundry_agent.py    Microsoft Foundry backend (needs Azure)
├── data/knowledge_base.txt Sample developer knowledge base (30 articles)
├── tests/
│   ├── fake_ollama.py               Scripted fake Ollama HTTP server   (new)
│   ├── test_ollama_agent.py         Unit tests: tool loop (fake client) (new)
│   ├── test_ollama_integration.py   Integration: HTTP, streaming, fallback, CLI
│   ├── test_ollama_live.py          Opt-in: real qwen3:4b must call each tool
│   ├── test_tools.py, test_agent.py, test_config.py, test_foundry_agent.py
├── .env / .env.example, .gitignore, requirements.txt, README.md
```

### What changed in the Ollama upgrade, and why

| Change | Reason |
|---|---|
| **`app/ollama_agent.py`** with `OllamaClient` and `OllamaAgentBackend` | Keeps the LLM provider in its own module, just as Foundry has its own. The client calls Ollama's REST API with `requests`, which was already a dependency, so **no new packages** and no Azure imports. |
| **Model-driven tool selection** | The model receives the three tool schemas and decides whether to call them. There's no regex in this path. |
| **Startup capability check** (`verify_tool_support`) | Calls `POST /api/show` and refuses models whose `capabilities` don't include `tools`. Compatibility is verified at runtime, not assumed. |
| **Backend-owned model context** (`OllamaAgentBackend.messages`) | Ollama needs exact `assistant.tool_calls` and `role: "tool"` messages. `ConversationHistory` stays the human-readable session transcript. |
| **Turn rollback** | If a turn fails mid-loop, its messages are removed, so the model never sees a question without an answer or a tool call without a result. |
| **Context trimming by whole turns** | Keeps the prompt within `MAX_HISTORY_MESSAGES` without separating a tool call from its result. |
| **`FallbackBackend`** in `agent.py` | If Ollama fails during a turn, the rule-based backend answers that turn and labels the reply. The LLM is tried again next turn. |
| **Startup fallback** in `create_backend()` | If Ollama is not running, the model is missing, or it lacks tool support, the app starts in rule-based mode with a warning. Set `OLLAMA_FALLBACK_TO_RULES=false` to fail instead. |
| **`ToolSpec.to_ollama_tool()`** | The same schemas in Ollama's nested `{"type":"function","function":{...}}` format. Validation still happens in `execute_tool()`. |
| **Extra system-prompt rules** | Small local models benefit from explicit rules: use only schema arguments, never invent tool results, retry once after a tool error. |
| **Default mode is now `ollama`** | `local` and `foundry` are unchanged and still selectable. |

### Fixes found by running the real `qwen3:4b` (Ollama 0.40.13)

The first version was only tested against a fake server. Running it against
the real model exposed three problems:

| Problem seen with the real model | Fix |
|---|---|
| The `qwen3:4b` tag is **Qwen3 Thinking 2507**, which always reasons first. With `think: false`, that reasoning and a dangling `</think>` leaked into the answer ("Okay, the user asked..."). | `OLLAMA_THINK` now defaults to `true`, so Ollama returns the reasoning in a separate `thinking` field that is never shown. `_clean_content()` also strips everything up to an unopened `</think>`. |
| On a CPU (~7 tokens/s), a turn takes 1–3 minutes, so the 120 s whole-request timeout fired. | `/api/chat` is now **streamed**. The timeout applies to gaps between chunks, so a slow but active generation never times out, while a stalled server still does. |
| Ollama loaded the model with a **4096-token** context, too small for tool schemas, search results and history. It silently drops the oldest tokens, starting with the system prompt. | New `OLLAMA_NUM_CTX` (default `8192`) is sent with every request. A warning is logged when a prompt reaches 90% of it. |

---

## 2. How the LLM tool-calling loop works (`ollama` mode)

For each user message, `OllamaAgentBackend.generate_reply()`:

1. Appends `{"role": "user", ...}` to the model context.
2. Sends `POST /api/chat` (streamed, `"stream": true`) with all messages
   (system prompt, earlier turns, earlier tool calls and results), the three
   **tool schemas**, `temperature`, `num_ctx` and `think`. The streamed chunks
   are merged into one message (`content`, `thinking`, `tool_calls`).
3. If the reply has **no** `tool_calls`, that text is the final answer.
4. Otherwise, for **each** tool call (models may request several at once):
   - looks the tool up in the registry (an unknown name produces an error result),
   - **validates arguments** against the JSON schema: missing, unexpected or
     wrong-type arguments and invalid JSON all produce error results,
   - **executes** the Python function. The calculator uses an AST whitelist, never `eval()`,
   - appends `{"role": "tool", "tool_name": ..., "content": "<JSON result>"}`.
5. Calls the model again with the results, and repeats from step 3.
   After 5 tool rounds without an answer, the turn fails with an error.

Tool errors are **returned to the model**, not raised. The model can read
`"ok": false, "error": "..."` and fix its arguments or explain the problem.

Errors handled:

| Failure | What happens |
|---|---|
| Ollama not running / connection refused | `OllamaConnectionError`, then the rule-based fallback is used (or the app exits with a clear message) |
| Model not downloaded | Message tells you to run `ollama pull <model>` |
| Model without tool support | Rejected at startup, with suggested models |
| Server sends nothing for `OLLAMA_TIMEOUT_SECONDS` (model loading, stalled) | `OllamaTimeoutError`. Slow but streaming generations do not time out |
| HTTP 4xx/5xx, invalid JSON, in-stream `error`, stream cut off, missing `message` | `OllamaResponseError` |
| Prompt near the context window | Warning: type `clear` or raise `OLLAMA_NUM_CTX` |
| Invalid/unknown/malformed tool calls | Error result sent back to the model; the session continues |
| Endless tool calling | Stops after 5 rounds |

---

## 3. Choosing a model

The model **must** support tool calling. On [ollama.com/library](https://ollama.com/library),
tool-capable models carry a **tools** tag, and `ollama show <model>` lists `tools` under *Capabilities*.

| Model | Download | Notes |
|---|---|---|
| `qwen3:4b` (default) | ~2.5 GB | Tagged *tools* and *thinking*. Currently Qwen3 **Thinking 2507**: it always reasons, so it is accurate but slow on a CPU |
| `llama3.2:3b` | ~2.0 GB | Tagged *tools*; lighter alternative for low-RAM machines |
| `qwen3:8b` | ~5.2 GB | Better answers if you have ≥16 GB RAM |

Sizes are from the Ollama library pages. Small models sometimes choose tools
poorly or answer without searching. Larger models are more reliable.

**Speed:** on the test laptop (8 CPU threads, 16 GB RAM, Intel Iris Xe, no
GPU offload) `qwen3:4b` generates about 7 tokens/s. A question that needs one
tool means two model calls, roughly 2–5 minutes in total, mostly reasoning.
A dedicated GPU is much faster.

---

## 4. Setup

Requirements: Python 3.13. For `ollama` mode, also Ollama itself (Windows, macOS or Linux).

### 4.1 Python environment

```powershell
cd foundry-agent-project
python -m venv .venv                     # skip if .venv already exists
.venv\Scripts\Activate.ps1               # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
Copy-Item .env.example .env              # skip if .env exists; macOS/Linux: cp
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 4.2 Install Ollama and a model

1. Install Ollama:
   - **Windows:** download the installer from <https://ollama.com/download>,
     or run `winget install Ollama.Ollama`.
   - **macOS:** download from <https://ollama.com/download>.
   - **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`
2. Make sure the server is running. The desktop app starts it automatically;
   otherwise run `ollama serve`. Check it with:
   ```powershell
   ollama --version
   curl http://localhost:11434/api/version
   ```
3. Download a tool-capable model and confirm it has the `tools` capability:
   ```powershell
   ollama pull qwen3:4b
   ollama show qwen3:4b          # "Capabilities" must include: tools
   ```
4. Optional quick test: `ollama run qwen3:4b "Say hello"`.

To use a different model, set `OLLAMA_MODEL` in `.env`.

---

## 5. Run

```powershell
python -m app.agent                  # Ollama LLM (default); falls back to rules if unavailable
python -m app.agent --mode local     # rule-based only, no LLM
python -m app.agent --mode ollama    # force Ollama mode regardless of .env

python -m unittest discover -s tests -t . -v   # 101 tests, no Ollama or Azure needed

$env:OLLAMA_LIVE_TESTS = "1"                     # opt-in: real model, slow
python -m unittest tests.test_ollama_live -v
Remove-Item Env:OLLAMA_LIVE_TESTS
```

CLI commands: `help`, `tools`, `history`, `clear`, `exit`.
The banner shows which backend is active. Logs go to `logs/agent.log`.

If Ollama isn't running you'll see:
```
WARNING: Cannot reach Ollama at http://localhost:11434. Start the Ollama app or run
'ollama serve', then try again. Using the rule-based fallback for this session.
```

### Real `ollama` run: `qwen3:4b` on Ollama 0.40.13 (actual output)

Output of `python -m unittest tests.test_ollama_live -v` on 2026-11-03
(5 tests passed in 699 s on a CPU-only laptop). The `[tool]` lines are calls
**the model chose** to make. Answers are shortened.

```
Q: Calculate (1200 * 0.18) + 50
[tool] calculate_expression({'expression': '(1200 * 0.18) + 50'}) -> ok=True
A: The result of `(1200 * 0.18) + 50` is **266**.

Q: Calculate 6 * 7.
[tool] calculate_expression({'expression': '6 * 7'}) -> ok=True
A: The result of $ 6 	imes 7 $ is **42**.

Q: Now multiply that result by 10.
[tool] calculate_expression({'expression': '42 * 10'}) -> ok=True      <- used history
A: The result of $ 42 	imes 10 $ is **420**.

Q: What is today's date?
[tool] get_current_datetime({}) -> ok=True
A: Today's date is November 3, 2026 (Tuesday).

Q: What is 17 * 23, and what day of the week is it today?
[tool] calculate_expression({'expression': '17 * 23'}) -> ok=True      <- two tools, one turn
[tool] get_current_datetime({}) -> ok=True
A: The result of 17 * 23 is **391**. Today is **Tuesday**.

Q: How do I fix a CORS error in my React app?
[tool] search_knowledge_base({'query': 'react cors error fix'}) -> ok=True
A: Here's how to fix CORS errors in your React app based on the knowledge base:
   1. Server-side solution: add Access-Control-Allow-Origin ... handle preflight OPTIONS ...
```

Wording varies between runs; the tests check the tool calls and tool-derived
values, not the exact text.

### Example session: `local` mode (actual output)

```
You: calculate (1200 * 0.18) + 50
  [tool] calculate_expression({"expression": "(1200 * 0.18) + 50"}) -> ok
Agent: (1200 * 0.18) + 50 = 266

You: calculate __import__("os").system("dir")
  [tool] calculate_expression(...) -> failed
Agent: I couldn't calculate that: Only these functions are allowed: abs, ceil, floor, log10, round, sqrt
```

---

## 6. Custom tools (`app/tools.py`)

| Tool | Input schema | Output |
|---|---|---|
| `get_current_datetime` | `{}` | `iso_8601, date, time, weekday, timezone, utc_offset` |
| `calculate_expression` | `{"expression": string}` | `{"expression", "result"}` |
| `search_knowledge_base` | `{"query": string}` | `{"query", "match_count", "results": [{category, title, content, score}]}` |

- Schemas are strict-mode compatible: every property is required and `additionalProperties` is false.
- `to_ollama_tool()`, `to_function_schema()` (Foundry/Responses) and the
  Foundry `FunctionTool` objects are all generated from the same `ToolSpec`.
- `execute_tool()` is the single validation and execution point for all backends. It never raises:
  it returns `{"ok": true, "result": ...}` or `{"ok": false, "error": ...}`.
- **Calculator:** parsed with `ast` and evaluated against a whitelist (numbers,
  `+ - * / // % **`, `pi`, `e`, `sqrt`, `abs`, `round`, `floor`, `ceil`,
  `log10`). It never uses `eval()`. Exponents, result size, `inf`/`nan` and division by zero are guarded.
- **Knowledge base search** is keyword retrieval with IDF weighting. It is
  **not production RAG**: no embeddings, no semantic matching, no vector index.

---

## 7. Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_MODE` | `ollama` | `ollama`, `local` or `foundry` (`--mode` overrides it) |
| `LOG_LEVEL` / `LOG_FILE` | `INFO` / `logs/agent.log` | Logging |
| `KNOWLEDGE_BASE_PATH` | `data/knowledge_base.txt` | Knowledge base file |
| `MAX_HISTORY_MESSAGES` | `50` | Session transcript size **and** LLM context size |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3:4b` | Must have the `tools` capability |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Max silence from the server, in seconds (1–600). Not a limit on total generation time |
| `OLLAMA_TEMPERATURE` | `0.2` | 0–2; lower gives steadier tool use |
| `OLLAMA_THINK` | `true` | Ollama `think` flag, only sent to models that support it. Keep `true` for `qwen3:4b`; `false` speeds up hybrid models such as the original `qwen3:8b` |
| `OLLAMA_NUM_CTX` | `8192` | Context window in tokens (2048–262144). More uses more RAM |
| `OLLAMA_FALLBACK_TO_RULES` | `true` | Use rule-based backend if Ollama fails |
| `FOUNDRY_*` | (empty) | Foundry mode only |

There are **no credentials** in this project. Ollama needs none, and Foundry uses Microsoft Entra ID.
`OLLAMA_BASE_URL` is used instead of Ollama's own `OLLAMA_HOST`, because that
server variable often holds a bind address such as `0.0.0.0`.

---

## 8. Tests

`python -m unittest discover -s tests -t . -v` runs 101 tests in about 15 seconds. They need **no Ollama, no Azure and no internet**.
The 5 live tests are skipped unless `OLLAMA_LIVE_TESTS=1`.

| File | What it covers |
|---|---|
| `test_tools.py` | Tools, safe calculator, knowledge base, schemas, argument validation |
| `test_ollama_agent.py` | Tool loop with a fake client: single, parallel and sequential tool calls, invalid arguments, unknown tools, malformed calls, round limit, rollback, context trimming, capability check |
| `test_ollama_integration.py` | Real HTTP against `tests/fake_ollama.py`: multi-turn workflow, streamed chunk merging, mid-stream stall vs. slow-but-alive stream, in-stream errors, connection refused, timeout, HTTP errors, invalid JSON, missing model, model without tools, startup and per-turn fallback, CLI subprocess run, no Azure imports |
| `test_ollama_live.py` | **Opt-in, real model.** `qwen3:4b` must call each tool itself, and the answer must contain values only the tool could produce: today's date, 266, the "CORS Errors" article, two tools in one turn (391 and the weekday), and a follow-up ("multiply that result by 10" gives 420) that needs conversation history |
| `test_agent.py` | Rule-based backend, orchestrator, CLI |
| `test_config.py` | Configuration validation for all modes |
| `test_foundry_agent.py` | Foundry request shapes with fake clients |

The default suite mocks the model, so it proves the **plumbing** is correct.
`test_ollama_live.py` checks that the real model actually calls the tools.

---

## 9. Ollama vs. Microsoft Foundry

| Concept | Ollama mode (this project) | Microsoft Foundry |
|---|---|---|
| Model hosting | Your machine | Azure model deployment |
| Agent definition | System prompt + tool list sent on every request | Versioned agent resource (`PromptAgentDefinition`) |
| Tool schema format | `{"type":"function","function":{...}}` | `FunctionTool(name, parameters, strict)` |
| Model requests a tool | `message.tool_calls[].function` | `function_call` output items |
| Tool result | `{"role":"tool","tool_name":...}` | `function_call_output` with `call_id` |
| Conversation state | Kept by the client (`OllamaAgentBackend.messages`) | Stored server-side (conversations) |
| Auth | None (local) | Microsoft Entra ID |

The core skill is the same: define tools with clear descriptions and schemas,
let the model choose, validate and execute safely, return results, and loop until done.

### Future Microsoft Foundry steps
When you have an Azure subscription: create a Foundry project, deploy a
tool-capable model, run `az login`, set `AGENT_MODE=foundry`,
`FOUNDRY_PROJECT_ENDPOINT` and `FOUNDRY_MODEL_NAME` in `.env`, then run
`python -m app.agent --mode foundry`. Next topics to explore: File Search or
Azure AI Search instead of the keyword knowledge base, Code Interpreter,
OpenAPI tools, tracing and evaluations.

---

## 10. Security notes

- No secrets in code or `.env`. `.env`, `.venv/`, `logs/` and `__pycache__/` are git-ignored.
- The calculator never uses `eval()`. All tool input from the model is validated against the schema.
- Tool errors returned to the model never include stack traces; details go to `logs/agent.log`.
- The log contains user questions and tool arguments, so treat it as local data.
- Ollama listens on localhost by default. Don't point `OLLAMA_BASE_URL` at an untrusted server.
