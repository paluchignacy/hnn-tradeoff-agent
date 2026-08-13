# hnn-tradeoff-agent

LangChain agent that reads Optuna training-run records from the
[hnn](https://github.com/paluchignacy/hnn) project's SQLite database
(`runs_minimal` table, via `scripts/db_reader.py`) and reasons about the
tradeoff between physical accuracy (Hamiltonian error/drift, trajectory
MSE) and numerical training cost (timestep, epochs, hidden-dim size,
etc.). See `hnn`'s [issue #4](https://github.com/paluchignacy/hnn/issues/4)
for the tracked motivation.

This repo is the agent only — the physics simulation, training runs, and
the `runs_minimal` schema all live in the `hnn` repo above. Point
`HNN_DB_PATH` at the SQLite file that `hnn` writes to.

## LLM backend

The agent talks to a local `qwen2.5:3b` model served by
[ollama-host](ollama-host/README.md) (Docker Compose + Ollama) at
`http://localhost:11434/v1`, using the OpenAI-compatible
`langchain_openai.ChatOpenAI` client — no `OPENAI_API_KEY` / paid API
required for local runs. `scripts/agent.py` builds the agent via
`langchain.agents.create_agent`.

## Progress log

- 2026-08-13: switched `scripts/agent.py` from `initialize_agent` /
  `AgentType.OPENAI_FUNCTIONS` (deprecated LangChain 0.x API) to
  `create_agent` with a `@tool`-decorated `summarize_hnn_runs`, and from
  OpenAI's `gpt-4o-mini` to the local `qwen2.5:3b` model via
  `ollama-host`.
- Ran a manual end-to-end query (`python scripts/agent.py "Summarize the
  tradeoff between hamiltonian_error and hnn_hidden_dim."`) against the
  local `hnn_results.sqlite` and the local qwen model. It completed
  successfully in ~4 minutes, correctly calling `summarize_hnn_runs` and
  returning a text summary of the tradeoff. The 3B model on CPU is slow
  on this host, especially while `hnn`'s `run_study.py` is also running
  and saturating CPU — that latency is from CPU-bound local inference
  under contention, not a bug in the agent.

## Layout

```
scripts/
  db_reader.py   read-only access to hnn's runs_minimal table
  agent.py       LangChain agent entrypoint
ollama-host/     local Ollama (qwen2.5:3b) via Docker Compose
```
