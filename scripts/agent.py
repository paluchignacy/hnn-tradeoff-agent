"""LangChain agent comparing physical vs numerical tradeoffs across HNN runs.

Wraps db_reader.load_runs() as a tool so an LLM can reason about tradeoffs
(e.g. hamiltonian_error vs hnn_hidden_dim, or timestep vs hamiltonian_drift)
across the Optuna trials recorded in hnn's runs_minimal table.
"""

import argparse

from db_reader import NUMERICAL_COLUMNS, PHYSICAL_COLUMNS, load_runs
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI


def summarize_runs(_: str) -> str:
    df = load_runs()
    if df.empty:
        return "No runs found in runs_minimal."
    cols = ["run_tag", "trial_number", *PHYSICAL_COLUMNS, *NUMERICAL_COLUMNS]
    cols = [c for c in cols if c in df.columns]
    return df[cols].describe().to_string()


def build_agent() -> AgentExecutor:
    tools = [
        Tool(
            name="summarize_hnn_runs",
            func=summarize_runs,
            description=(
                "Returns summary statistics (count/mean/std/min/max) for physical "
                "columns (hamiltonian_error, hamiltonian_drift, trajectory_mse_max, "
                "E/B fields, init velocities) and numerical columns (timestep, "
                "epochs, learning_rate, hnn_hidden_dim, final_test_loss_mse, "
                "time_to_tolerance) across all recorded HNN training runs."
            ),
        )
    ]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)


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
    print(agent.run(args.question))


if __name__ == "__main__":
    main()
