"""LangChain agent comparing physical vs numerical tradeoffs across HNN runs.

Wraps db_reader.load_runs() as a tool so an LLM can reason about tradeoffs
(e.g. hamiltonian_error vs hnn_hidden_dim, or timestep vs hamiltonian_drift)
across the Optuna trials recorded in hnn's runs_minimal table.
"""

import argparse

from db_reader import NUMERICAL_COLUMNS, PHYSICAL_COLUMNS, load_runs
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def summarize_hnn_runs(_: str = "") -> str:
    """Returns summary statistics (count/mean/std/min/max) for physical
    columns (hamiltonian_error, hamiltonian_drift, trajectory_mse_max,
    E/B fields, init velocities) and numerical columns (timestep, epochs,
    learning_rate, hnn_hidden_dim, final_test_loss_mse, time_to_tolerance)
    across all recorded HNN training runs."""
    df = load_runs()
    if df.empty:
        return "No runs found in runs_minimal."
    cols = ["run_tag", "trial_number", *PHYSICAL_COLUMNS, *NUMERICAL_COLUMNS]
    cols = [c for c in cols if c in df.columns]
    return df[cols].describe().to_string()


def build_agent():
    llm = ChatOpenAI(
        model="qwen2.5:3b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
    )
    return create_agent(model=llm, tools=[summarize_hnn_runs])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default=(
            "Compare the physical accuracy (hamiltonian_error, hamiltonian_drift) "
            "against the numerical training cost (epochs, hnn_hidden_dim) across "
            "runs, and summarize the tradeoff."
        ),
    )
    args = parser.parse_args()
    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(args.question)]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
