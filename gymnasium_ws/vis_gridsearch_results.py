"""
vis_gridsearch.py — Visualize hyperparameter grid search results.

Usage:
    python3 vis_gridsearch.py path/to/grid_search_results.csv
    python3 vis_gridsearch.py path/to/grid_search_results.csv --algo sac
"""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def plot_algo(df: pd.DataFrame, algo: str, out_dir: Path):
    data = df[df["algo"] == algo]
    if data.empty:
        print(f"No data for algo '{algo}', skipping.")
        return

    params = [c for c in data.columns if c not in ("algo", "seed", "final_mean_reward", "steps")]

    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 4), sharey=True)
    if len(params) == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        sns.stripplot(data=data, x=param, y="final_mean_reward", ax=ax,
                      jitter=True, alpha=0.6, size=6)
        means = data.groupby(param)["final_mean_reward"].mean()
        for i, val in enumerate(means.values):
            ax.hlines(val, i - 0.2, i + 0.2, colors="red", linewidths=2)
        ax.set_title(param)
        ax.set_xlabel("")

    axes[0].set_ylabel("final mean reward")
    axes[0].legend(handles=[plt.Line2D([0], [0], color="red", linewidth=2, label="mean")],
                   loc="lower right")
    fig.suptitle(f"{algo.upper()} hyperparameter grid search")
    plt.tight_layout()

    out_path = out_dir / f"grid_search_{algo}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize hyperparameter grid search results")
    parser.add_argument("csv", type=Path, help="Path to grid_search_results.csv")
    parser.add_argument("--algo", type=str, choices=["sac", "ppo"],
                        help="Only plot this algo (default: plot all present)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = args.csv.parent

    algos = [args.algo] if args.algo else df["algo"].unique().tolist()
    for algo in algos:
        plot_algo(df, algo, out_dir)


if __name__ == "__main__":
    main()