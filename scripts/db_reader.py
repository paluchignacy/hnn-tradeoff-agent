"""Read-only access to the hnn project's runs_minimal SQLite table."""

import os
import sqlite3

import pandas as pd

DB_PATH = os.environ.get(
    "HNN_DB_PATH", "/home/jovyan/mgr/hnn_results_rk45-onlyele.sqlite"
)

PHYSICAL_COLUMNS = [
    "hamiltonian_error",
    "hamiltonian_drift",
    "hamiltonian_drift_pred",
    "trajectory_mse_max",
    "E_x",
    "E_y",
    "E_z",
    "B_x",
    "B_y",
    "B_z",
    "init_vx",
    "init_vy",
    "init_vz",
]

NUMERICAL_COLUMNS = [
    "t_end",
    "timestep",
    "epochs",
    "learning_rate",
    "batch_size",
    "optimizer",
    "hnn_hidden_dim",
    "final_test_loss_mse",
    "time_to_tolerance",
    "hamiltonian_loss_weight",
]


def load_runs() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        return pd.read_sql_query("SELECT * FROM runs_minimal", con)
    finally:
        con.close()
