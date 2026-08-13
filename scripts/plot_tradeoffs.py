"""Generates visual tradeoff benchmarks from hnn's runs_minimal table.

Produces two PNGs:
  - correlation_heatmap.png: physical columns x numerical columns,
    Pearson correlation.
  - top_tradeoffs.png: scatter plots for the most strongly correlated
    physical/numerical pairs (physical accuracy vs numerical cost).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from db_reader import NUMERICAL_COLUMNS, PHYSICAL_COLUMNS, load_runs


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    physical = [c for c in PHYSICAL_COLUMNS if c in df.columns]
    numerical = [
        c for c in NUMERICAL_COLUMNS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[physical + numerical].corr(numeric_only=True).loc[physical, numerical]


def plot_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.2 * len(corr.columns) + 2, 0.6 * len(corr.index) + 2))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Physical accuracy vs numerical cost: Pearson correlation")
    fig.colorbar(im, ax=ax, label="correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_tradeoffs(df: pd.DataFrame, corr: pd.DataFrame, out_path: Path, top_n: int = 4) -> None:
    pairs = corr.stack().dropna().rename("correlation").reset_index()
    pairs.columns = ["physical_column", "numerical_column", "correlation"]
    pairs = pairs.reindex(pairs["correlation"].abs().sort_values(ascending=False).index)
    top = pairs.head(top_n)

    fig, axes = plt.subplots(1, len(top), figsize=(5 * len(top), 4), squeeze=False)
    for ax, (_, row) in zip(axes[0], top.iterrows()):
        x, y = row["numerical_column"], row["physical_column"]
        ax.scatter(df[x], df[y], alpha=0.7)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"r = {row['correlation']:.2f}")
    fig.suptitle("Strongest physical/numerical tradeoffs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", default="plots", help="Directory to write PNGs into (default: plots)"
    )
    parser.add_argument(
        "--top-n", type=int, default=4, help="Number of scatter plots in top_tradeoffs.png"
    )
    args = parser.parse_args()

    df = load_runs()
    if df.empty:
        raise SystemExit("No runs found in runs_minimal.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    corr = correlation_matrix(df)
    plot_heatmap(corr, outdir / "correlation_heatmap.png")
    plot_top_tradeoffs(df, corr, outdir / "top_tradeoffs.png", top_n=args.top_n)
    print(f"Wrote {outdir / 'correlation_heatmap.png'} and {outdir / 'top_tradeoffs.png'}")


if __name__ == "__main__":
    main()
