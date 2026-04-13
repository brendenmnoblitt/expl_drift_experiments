"""Generate figures for the transformer drift detection paper.

Produces:
  1. drift_curves.png         — per-window drift metric with mean±std over seeds
  2. alert_timeline.png       — heatmap of alert levels per seed per window
  3. accuracy_vs_drift.png    — dual-axis plot showing drift rises while acc stays flat
  4. architecture_comparison.png — BERT vs GPT-2 first-warning window distributions

Usage:
    PYTHONPATH=/home/brendenadm/projects python experiments/transformer_drift/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.transformer_drift.config import (
    DRIFT_START_WINDOW,
    DRIFT_TYPES,
    N_CALIBRATION,
    RESULTS_DIR,
)

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BERT_RESULTS = RESULTS_DIR / "run_bert_v2_20260412_135332"
GPT2_RESULTS = RESULTS_DIR / "run_gpt2_v2_20260412_143738"

DRIFT_LABELS = {
    "domain_shift": "Domain Shift",
    "vocabulary_shift": "Vocabulary Shift",
    "class_distribution_shift": "Class Distribution Shift",
}

MODEL_COLORS = {"bert": "#1f77b4", "gpt2": "#d62728"}
MODEL_LABELS = {"bert": "BERT (encoder)", "gpt2": "GPT-2 (decoder-only)"}


def load_per_window() -> pd.DataFrame:
    bert = pd.read_csv(BERT_RESULTS / "per_window_results.csv")
    gpt2 = pd.read_csv(GPT2_RESULTS / "per_window_results.csv")
    return pd.concat([bert, gpt2], ignore_index=True)


def load_summary() -> pd.DataFrame:
    bert = pd.read_csv(BERT_RESULTS / "summary.csv")
    gpt2 = pd.read_csv(GPT2_RESULTS / "summary.csv")
    return pd.concat([bert, gpt2], ignore_index=True)


MODEL_METRIC = {"bert": "cosine_drift", "gpt2": "max_wasserstein"}


def figure_drift_curves(df: pd.DataFrame) -> None:
    """Per-window drift metric (model's native metric), mean±std across seeds."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)

    for row, model in enumerate(["bert", "gpt2"]):
        metric = MODEL_METRIC[model]
        for col, drift in enumerate(DRIFT_TYPES):
            ax = axes[row, col]
            sub = df[
                (df.model_type == model)
                & (df.drift_type == drift)
                & df[metric].notna()
            ]
            grouped = sub.groupby("window_id")[metric].agg(["mean", "std"])
            x = grouped.index.values
            mean = grouped["mean"].values
            std = grouped["std"].values
            color = MODEL_COLORS[model]
            ax.plot(x, mean, color=color, lw=2)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

            ax.axvline(DRIFT_START_WINDOW, color="black", linestyle="--", lw=1)
            ax.axvline(1 + N_CALIBRATION, color="gray", linestyle=":", lw=1)

            if row == 0:
                ax.set_title(DRIFT_LABELS[drift], fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\n{metric}", fontsize=10)
            if row == 1:
                ax.set_xlabel("Window", fontsize=11)

            ax.grid(True, alpha=0.3)

    fig.suptitle("Attribution Drift Metrics Over Time (mean ± std across 10 seeds)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out = FIGURES_DIR / "drift_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_alert_shaded_curves(df: pd.DataFrame) -> None:
    """Per-seed drift curves with window shading by alert level (yellow=warn, red=critical)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    alert_colors = {"warning": "#ffeb99", "critical": "#ff9999"}

    for row, model in enumerate(["bert", "gpt2"]):
        metric = MODEL_METRIC[model]
        for col, drift in enumerate(DRIFT_TYPES):
            ax = axes[row, col]
            sub = df[
                (df.model_type == model)
                & (df.drift_type == drift)
                & df[metric].notna()
            ]

            # Shading from alert counts per window (across seeds)
            for wid in sorted(sub.window_id.unique()):
                window_rows = sub[sub.window_id == wid]
                n_total = len(window_rows)
                if n_total == 0:
                    continue
                n_crit = (window_rows.alert_level == "critical").sum()
                n_warn = (window_rows.alert_level == "warning").sum()
                frac_crit = n_crit / n_total
                frac_warn = n_warn / n_total
                # Dominant alert color weighted by fraction
                if frac_crit > 0.5:
                    ax.axvspan(wid - 0.5, wid + 0.5, color=alert_colors["critical"], alpha=0.6)
                elif frac_warn + frac_crit > 0.5:
                    ax.axvspan(wid - 0.5, wid + 0.5, color=alert_colors["warning"], alpha=0.6)

            # Plot each seed lightly, plus the mean
            for seed in sub.seed.unique():
                seed_data = sub[sub.seed == seed].sort_values("window_id")
                ax.plot(seed_data.window_id, seed_data[metric],
                        color=MODEL_COLORS[model], alpha=0.25, lw=1)
            mean_curve = sub.groupby("window_id")[metric].mean()
            ax.plot(mean_curve.index, mean_curve.values,
                    color=MODEL_COLORS[model], lw=2.5, label="mean")

            ax.axvline(DRIFT_START_WINDOW, color="black", linestyle="--", lw=1)

            if row == 0:
                ax.set_title(DRIFT_LABELS[drift], fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\n{metric}", fontsize=10)
            if row == 1:
                ax.set_xlabel("Window", fontsize=11)
            ax.grid(True, alpha=0.3)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=alert_colors["warning"], alpha=0.6, label="majority warning"),
        Patch(facecolor=alert_colors["critical"], alpha=0.6, label="majority critical"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle("Drift Curves with Alert-Level Window Shading", fontsize=13, y=1.04)
    fig.tight_layout()
    out = FIGURES_DIR / "alert_shaded_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_alert_timeline(df: pd.DataFrame) -> None:
    """Heatmap: rows=seed, cols=window, color=alert level. One panel per model×drift."""
    alert_to_val = {"ok": 0, "warning": 1, "critical": 2}
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True, sharey=True)

    for row, model in enumerate(["bert", "gpt2"]):
        for col, drift in enumerate(DRIFT_TYPES):
            ax = axes[row, col]
            sub = df[
                (df.model_type == model)
                & (df.drift_type == drift)
                & df.alert_level.notna()
            ]
            pivot = sub.pivot_table(
                index="seed", columns="window_id", values="alert_level",
                aggfunc="first",
            )
            numeric = pivot.map(lambda a: alert_to_val.get(a, np.nan))

            im = ax.imshow(
                numeric.values, aspect="auto", cmap="RdYlGn_r",
                vmin=0, vmax=2, interpolation="nearest",
            )
            ax.set_xticks(range(len(numeric.columns)))
            ax.set_xticklabels(numeric.columns, fontsize=8)
            ax.set_yticks(range(len(numeric.index)))
            ax.set_yticklabels(numeric.index, fontsize=8)

            ax.axvline(
                DRIFT_START_WINDOW - numeric.columns[0] - 0.5,
                color="black", lw=2,
            )

            if row == 0:
                ax.set_title(DRIFT_LABELS[drift], fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nseed", fontsize=11)
            if row == 1:
                ax.set_xlabel("Window", fontsize=11)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["ok", "warning", "critical"])

    fig.suptitle("Alert Level per Seed per Window", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    out = FIGURES_DIR / "alert_timeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_accuracy_vs_drift(df: pd.DataFrame) -> None:
    """Dual-axis: accuracy flat while drift rises — the core finding."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    for row, model in enumerate(["bert", "gpt2"]):
        for col, drift in enumerate(DRIFT_TYPES):
            ax = axes[row, col]
            sub = df[(df.model_type == model) & (df.drift_type == drift)]

            acc = sub.groupby("window_id")["accuracy"].agg(["mean", "std"])
            ax.plot(acc.index, acc["mean"], color="black", lw=2, label="accuracy")
            ax.fill_between(acc.index, acc["mean"] - acc["std"], acc["mean"] + acc["std"],
                            color="black", alpha=0.15)
            ax.set_ylim(0.85, 1.0)
            ax.set_ylabel("accuracy", color="black", fontsize=10)

            metric = "cosine_drift" if model == "bert" else "max_wasserstein"
            drift_sub = sub[sub[metric].notna()]
            if len(drift_sub):
                d = drift_sub.groupby("window_id")[metric].agg(["mean", "std"])
                ax2 = ax.twinx()
                color = MODEL_COLORS[model]
                ax2.plot(d.index, d["mean"], color=color, lw=2, label=metric)
                ax2.fill_between(d.index, d["mean"] - d["std"], d["mean"] + d["std"],
                                 color=color, alpha=0.2)
                ax2.set_ylabel(f"{metric}", color=color, fontsize=10)
                ax2.tick_params(axis="y", colors=color)

            ax.axvline(DRIFT_START_WINDOW, color="gray", linestyle="--", lw=1)

            if row == 0:
                ax.set_title(DRIFT_LABELS[drift], fontsize=12)
            if row == 1:
                ax.set_xlabel("Window", fontsize=11)

            if col == 0:
                ax.text(-0.22, 0.5, MODEL_LABELS[model], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=12, weight="bold")

            ax.grid(True, alpha=0.3)

    fig.suptitle("Accuracy Stays Flat While Attribution Drift Rises", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "accuracy_vs_drift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_first_warning_distribution(summary: pd.DataFrame) -> None:
    """Distribution of first-warning-window across seeds: per model × drift."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for col, drift in enumerate(DRIFT_TYPES):
        ax = axes[col]
        data = {
            model: summary[
                (summary.model_type == model) & (summary.drift_type == drift)
            ]["first_warning_window"].dropna().values
            for model in ["bert", "gpt2"]
        }

        positions = [1, 2]
        parts = ax.boxplot(
            [data["bert"], data["gpt2"]],
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showmeans=True,
        )
        for patch, model in zip(parts["boxes"], ["bert", "gpt2"]):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.6)

        ax.axhline(DRIFT_START_WINDOW, color="black", linestyle="--", lw=1,
                   label="drift injected" if col == 0 else None)

        ax.set_xticks(positions)
        ax.set_xticklabels([MODEL_LABELS["bert"], MODEL_LABELS["gpt2"]], fontsize=10)
        ax.set_title(DRIFT_LABELS[drift], fontsize=12)
        if col == 0:
            ax.set_ylabel("First warning window", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("First Alert Window: BERT vs GPT-2", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "architecture_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def aggregate_statistics(summary: pd.DataFrame, per_window: pd.DataFrame) -> pd.DataFrame:
    """Build a per-(model, drift_type) table of the core statistics for the paper."""
    rows = []
    for model in ["bert", "gpt2"]:
        for drift in DRIFT_TYPES:
            s = summary[(summary.model_type == model) & (summary.drift_type == drift)]
            pw = per_window[
                (per_window.model_type == model)
                & (per_window.drift_type == drift)
                & per_window.alert_level.notna()
            ]

            # Per-seed first-alert window
            first_alerts = (
                pw[pw.alert_level != "ok"]
                .groupby("seed")["window_id"].min()
            )
            # Detection rate: fraction of seeds with at least one warning/critical
            n_seeds = s.seed.nunique()
            detection_rate = len(first_alerts) / n_seeds if n_seeds else float("nan")

            # Alerts per window after drift start (rate of alerts)
            post_drift = pw[pw.window_id >= DRIFT_START_WINDOW]
            alert_rate = (post_drift.alert_level != "ok").mean()

            rows.append({
                "model": model,
                "drift_type": drift,
                "n_seeds": n_seeds,
                "detection_rate": round(detection_rate, 2),
                "first_alert_mean": round(first_alerts.mean(), 2) if len(first_alerts) else None,
                "first_alert_std": round(first_alerts.std(), 2) if len(first_alerts) > 1 else None,
                "post_drift_alert_rate": round(alert_rate, 2),
                "n_warnings_mean": round(s.n_warnings.mean(), 2),
                "n_criticals_mean": round(s.n_criticals.mean(), 2),
                "baseline_acc": round(s.baseline_accuracy.mean(), 3),
                "final_acc": round(s.final_accuracy.mean(), 3),
                "lead_time_detected": int(s.lead_time.notna().sum()),
                "lead_time_mean": round(s.lead_time.mean(), 2) if s.lead_time.notna().any() else None,
            })
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Loading results...")
    df = load_per_window()
    summary = load_summary()
    print(f"  {len(df)} per-window rows, {len(summary)} summary rows")

    print("Generating figures...")
    figure_drift_curves(df)
    figure_alert_shaded_curves(df)
    figure_alert_timeline(df)
    figure_accuracy_vs_drift(df)
    figure_first_warning_distribution(summary)

    print("Building aggregate statistics...")
    stats = aggregate_statistics(summary, df)
    stats_csv = FIGURES_DIR / "aggregate_statistics.csv"
    stats_md = FIGURES_DIR / "aggregate_statistics.md"
    stats.to_csv(stats_csv, index=False)
    # Manual markdown table (avoid tabulate dep)
    cols = list(stats.columns)
    with open(stats_md, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in stats.iterrows():
            f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
    print(f"  wrote {stats_csv}")
    print(f"  wrote {stats_md}")
    print()
    print(stats.to_string(index=False))
    print(f"\nAll artifacts saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
