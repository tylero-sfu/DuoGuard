"""Plotting helpers for public DuoGuard diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_first_k_hit_probability(summary: pd.DataFrame, out_path: str | Path) -> None:
    """Plot hit probability by shot position and weapon category."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (label, weapon), part in summary.groupby(["label_name", "weapon_category"]):
        part = part.sort_values("shot_position")
        ax.plot(part["shot_position"], part["hit_probability"], marker="o", label=f"{label} / {weapon}")
    ax.set_xlabel("Shot position in sequence")
    ax.set_ylabel("Hit probability")
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_minimum_sequence_sensitivity(results: pd.DataFrame, out_path: str | Path) -> None:
    """Plot player-level metric changes across minimum sequence thresholds."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for metric in ["auprc", "f1", "precision", "recall"]:
        if metric in results:
            ax.plot(results["n_min"], results[metric], marker="o", label=metric.upper())
    ax.set_xlabel("Minimum retained shooting sequences per player")
    ax.set_ylabel("Metric")
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

