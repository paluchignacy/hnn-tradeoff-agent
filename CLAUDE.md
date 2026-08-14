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

## Tools

- `summarize_hnn_runs` — count/mean/std/min/max over physical and
  numerical columns (`df.describe()`).
- `correlate_physical_vs_numerical` — Pearson correlation between each
  physical column (hamiltonian_error, hamiltonian_drift,
  trajectory_mse_max, E/B fields, init velocities) and each numeric
  training-cost column (timestep, epochs, learning_rate, batch_size,
  hnn_hidden_dim, hamiltonian_loss_weight), sorted by absolute
  correlation strength. Columns with zero variance in the current data
  (e.g. `t_end`, `timestep`, `hnn_hidden_dim` are fixed across all 63
  runs in the local test DB) correctly come back as `NaN`.

## Visual benchmarks

`scripts/plot_tradeoffs.py` renders the same physical/numerical
tradeoff data as PNGs (default output dir `plots/`, gitignored):
`correlation_heatmap.png` (physical x numerical Pearson correlation)
and `top_tradeoffs.png` (scatter plots for the strongest-correlated
pairs). This is a standalone script, not an agent tool — qwen is a text
model and can't view images, so plotting stays outside the LangChain
tool loop and is meant to be run/viewed directly by a human.

## CI

`.github/workflows/ci.yml` runs on push/PR to `main`: installs
`requirements.txt`, runs `ruff check scripts`, and imports
`agent`/`db_reader`/`plot_tradeoffs` to catch import-time breakage. No
test suite exists yet, so this is intentionally lint + import-only —
`agent.py`'s `build_agent()`/`load_runs()` aren't exercised since CI has
no DB file or local ollama server.

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
- Added `correlate_physical_vs_numerical` for actual data analysis
  (beyond describe()-style summaries). Verified the correlation math
  directly against the local DB, then ran it end-to-end through qwen
  ("Which numerical training settings correlate most strongly with
  physical accuracy?") — the model picked the new tool correctly and
  summarized the strongest correlations (e.g. `trajectory_mse_max` vs
  `final_test_loss_mse` at 0.98).
- Added `scripts/plot_tradeoffs.py` for visual benchmarks (correlation
  heatmap + top-tradeoff scatter plots), since text summaries don't show
  shape/outliers the way a plot does. Ran it against the local DB and
  visually confirmed both PNGs render correctly — e.g. the heatmap
  correctly leaves `t_end`/`timestep`/`hnn_hidden_dim` blank (constant
  in this dataset, so correlation is undefined) instead of erroring.
- 2026-08-14: added `.github/workflows/ci.yml` (lint + import check) —
  there was previously no CI at all, hence nothing showing up under the
  repo's Actions tab. Verified `ruff check scripts` passes clean and the
  import check succeeds locally before pushing.

## Layout

```
scripts/
  db_reader.py   read-only access to hnn's runs_minimal table
  agent.py       LangChain agent entrypoint
ollama-host/     local Ollama (qwen2.5:3b) via Docker Compose
```
