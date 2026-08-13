# hnn-tradeoff-agent

LangChain-based agent that reads training-run records from the
[hnn](https://github.com/paluchignacy/hnn) project's SQLite database
(`runs_minimal` table) and compares physical accuracy (Hamiltonian
error/drift, trajectory MSE) against numerical training cost (timestep,
epochs, hidden-dim size, etc.) across recorded Optuna trials.

Tracks https://github.com/paluchignacy/hnn/issues/4.

## Layout

```
scripts/
  db_reader.py       read-only access to hnn's runs_minimal table
  agent.py           LangChain agent entrypoint
  plot_tradeoffs.py  generates visual tradeoff benchmarks (PNGs)
```

## Setup

```
pip install -r requirements.txt
```

Point at the hnn SQLite DB (same file `HNN_DB_PATH` writes to in the hnn
repo) and provide an OpenAI key:

```
export HNN_DB_PATH=/path/to/hnn_results.sqlite
export OPENAI_API_KEY=...
```

## Run

```
python scripts/agent.py "your question about run tradeoffs"
```

## Visual benchmarks

Generate a correlation heatmap and scatter plots for the strongest
physical/numerical tradeoffs (written to `plots/` by default):

```
python scripts/plot_tradeoffs.py
```
