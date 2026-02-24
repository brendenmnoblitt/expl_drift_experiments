"""Plotting functions for experiment visualization."""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


def _ensure_dir(path: str) -> None:
    """Ensure the directory for the given path exists.
    
    Args:
        path: Path to the file to ensure directory exists for.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_drift_timeseries(
    drift_df: pd.DataFrame,
    accuracy_series: list[float] | np.ndarray,
    drift_start: int,
    title: str,
    save_path: str,
) -> None:
    """Plot drift metrics and accuracy over windows. Marks drift start and shows both
    signals on the same plot.
    
    Args:
        drift_df: DataFrame with drift metrics per window (indexed by window number).
        accuracy_series: List or array of accuracy values per window.
        drift_start: Window index where drift injection begins.
        title: Title for the plot.
        save_path: File path to save the figure.
    """
    _ensure_dir(save_path)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    for col in drift_df.columns:
        ax1.plot(drift_df.index, drift_df[col], marker="o", label=col, markersize=4)

    ax1.set_xlabel("Window")
    ax1.set_ylabel("Drift Metric")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(range(len(accuracy_series)), accuracy_series, color="red",
             linewidth=2, linestyle="--", label="Accuracy")
    ax2.set_ylabel("Accuracy", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right")

    ax1.axvline(x=drift_start, color="gray", linestyle=":", alpha=0.7, label="Drift Start")
    ax1.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_explanation_vs_data_drift(
    explanation_drift_df: pd.DataFrame,
    data_drift_df: pd.DataFrame,
    save_path: str,
) -> None:
    """Side-by-side comparison of explanation drift vs data drift. Plots all metrics from both
    DataFrames on the same figure with two subplots.
    
    Args:
        explanation_drift_df: DataFrame with explanation drift metrics per window.
        data_drift_df: DataFrame with data drift metrics per window.
        save_path: File path to save the figure.
    """
    _ensure_dir(save_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for col in explanation_drift_df.columns:
        ax1.plot(explanation_drift_df.index, explanation_drift_df[col],
                 marker="o", label=col, markersize=4)
    ax1.set_title("Explanation Drift")
    ax1.set_xlabel("Window")
    ax1.set_ylabel("Metric Value")
    ax1.legend(fontsize=8)

    for col in data_drift_df.columns:
        ax2.plot(data_drift_df.index, data_drift_df[col],
                 marker="s", label=col, markersize=4)
    ax2.set_title("Data Drift")
    ax2.set_xlabel("Window")
    ax2.set_ylabel("Metric Value")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_attribution_heatmap(
    attributions_per_window: list[np.ndarray],
    feature_names: list[str],
    save_path: str,
) -> None:
    """Heatmap of mean attribution per feature per window. Shows how feature importance patterns
    evolve over time.
    
    Args:
        attributions_per_window: List of arrays of shape (n_samples, n_features) with attributions
            for each window.
        feature_names: List of feature names corresponding to the columns in the attribution arrays.
        save_path: File path to save the figure.
    """
    _ensure_dir(save_path)
    mean_attrs = np.array([np.mean(np.abs(a), axis=0) for a in attributions_per_window])
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(mean_attrs, xticklabels=feature_names,
                yticklabels=[f"W{i}" for i in range(len(attributions_per_window))],
                cmap="YlOrRd", ax=ax)
    ax.set_title("Mean |Attribution| per Feature per Window")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Window")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_method_consistency(
    shap_drift: np.ndarray,
    lime_drift: np.ndarray,
    ig_drift: np.ndarray,
    save_path: str,
) -> None:
    """Correlation between explanation methods' drift signals. Plots pairwise scatter plots of
    drift metrics from different methods with correlation coefficients.
    
    Args:
        shap_drift: Array of drift metric values per window for SHAP.
        lime_drift: Array of drift metric values per window for LIME.
        ig_drift: Array of drift metric values per window for IG.
        save_path: File path to save the figure.
    """
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pairs = [
        (shap_drift, lime_drift, "SHAP", "LIME"),
        (shap_drift, ig_drift, "SHAP", "IG"),
        (lime_drift, ig_drift, "LIME", "IG"),
    ]

    for ax, (d1, d2, n1, n2) in zip(axes, pairs):
        min_len = min(len(d1), len(d2))
        ax.scatter(d1[:min_len], d2[:min_len], alpha=0.7)
        corr = np.corrcoef(d1[:min_len], d2[:min_len])[0, 1]
        ax.set_title(f"{n1} vs {n2} (r={corr:.3f})")
        ax.set_xlabel(n1)
        ax.set_ylabel(n2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_detection_comparison(
    drift_df: pd.DataFrame,
    accuracy_series: list[float] | np.ndarray,
    data_drift_df: pd.DataFrame,
    drift_start: int,
    metric_name: str,
    save_path: str,
) -> None:
    """Plot explanation drift vs data drift lead times for a specific metric.

    Shows both signals on the same plot with detection points marked.

    Args:
        drift_df: DataFrame with explanation drift metrics per window (indexed by window number).
        accuracy_series: List or array of accuracy values per window.
        data_drift_df: DataFrame with data drift metrics per window (indexed by window number).
        drift_start: Window index where drift injection begins.
        metric_name: Name of the drift metric to compare (e.g. "max_jsd", "energy_distance").
        save_path: File path to save the figure.    
    """
    _ensure_dir(save_path)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    if metric_name in drift_df.columns:
        expl_series = drift_df[metric_name]
        ax1.plot(expl_series.index, expl_series.values, marker="o",
                 label=f"Explanation: {metric_name}", color="blue", markersize=5)

    data_metric_map: dict[str, str] = {
        "max_jsd": "data_max_jsd",
        "jsd": "data_jsd",
        "energy_distance": "data_energy",
        "ks_max_statistic": "data_ks_max",
    }
    data_col = data_metric_map.get(metric_name)
    if data_col and data_col in data_drift_df.columns:
        data_series = data_drift_df[data_col]
        ax1.plot(data_series.index, data_series.values, marker="s",
                 label=f"Data: {data_col}", color="green", markersize=5)

    ax1.set_xlabel("Window")
    ax1.set_ylabel("Drift Metric")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(range(len(accuracy_series)), accuracy_series, color="red",
             linewidth=2, linestyle="--", label="Accuracy")
    ax2.set_ylabel("Accuracy", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right")

    ax1.axvline(x=drift_start, color="gray", linestyle=":", alpha=0.7)
    ax1.set_title(f"Explanation vs Data Drift: {metric_name}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_severity_sweep(
    severity_results: dict[float, dict[str, int | None]], save_path: str
) -> None:
    """Plot detection lead times across severity levels.

    severity_results: dict of {severity: {metric: lead_time}}
    None values (no detection) are omitted from the line and annotated with "x".
    
    Args:
        severity_results: Dictionary mapping severity levels to dictionaries of metric lead times.
        save_path: File path to save the figure.
    """
    _ensure_dir(save_path)
    severities = sorted(severity_results.keys())
    metrics = list(next(iter(severity_results.values())).keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    no_detect_severities: set[float] = set()

    for metric in metrics:
        detected_sev: list[float] = []
        detected_lt: list[int] = []
        for s in severities:
            lt = severity_results[s].get(metric)
            if lt is not None:
                detected_sev.append(s)
                detected_lt.append(lt)
            else:
                no_detect_severities.add(s)
        if detected_sev:
            ax.plot(detected_sev, detected_lt, marker="o", label=metric)

    for s in sorted(no_detect_severities):
        ax.annotate("No detection", xy=(s, 0), xytext=(s, 0.6),
                    ha="center", fontsize=9, color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
        ax.scatter([s], [0], marker="x", color="gray", s=80, zorder=5)

    ax.set_xlabel("Drift Severity (additive shift per window)")
    ax.set_ylabel("Detection Lead Time (windows ahead of accuracy drop)")
    ax.set_title("Detection Capability vs Drift Severity")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_multi_seed_drift(
    mean_df: pd.DataFrame,
    ci_lower_df: pd.DataFrame,
    ci_upper_df: pd.DataFrame,
    accuracy_mean: np.ndarray,
    accuracy_ci_lower: np.ndarray,
    accuracy_ci_upper: np.ndarray,
    drift_start: int,
    title: str,
    save_path: str,
) -> None:
    """Plot multi-seed drift metrics with an explicit accuracy-failure reference.

    Includes:
    - drift metric means +/- 95% CI
    - accuracy mean +/- 95% CI (secondary axis)
    - optional accuracy-drop threshold (mean-2*std from pre-drift windows)
    - first detected accuracy-drop marker (matching lead-time convention)

    Args:
        mean_df: DataFrame with mean drift metrics per window (indexed by window number).
        ci_lower_df: DataFrame with lower bound of 95% CI for drift metrics per window.
        ci_upper_df: DataFrame with upper bound of 95% CI for drift metrics per window.
        accuracy_mean: Array of mean accuracy values per window.
        accuracy_ci_lower: Array of lower bound of 95% CI for accuracy per window.
        accuracy_ci_upper: Array of upper bound of 95% CI for accuracy per window.
        drift_start: Window index where drift injection begins.
        title: Title for the plot.
        save_path: File path to save the figure.
    """
    _ensure_dir(save_path)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10.colors
    for idx, col in enumerate(mean_df.columns):
        color = colors[idx % len(colors)]
        ax1.plot(mean_df.index, mean_df[col], marker="o", label=col,
                 markersize=4, color=color)
        ax1.fill_between(mean_df.index, ci_lower_df[col], ci_upper_df[col],
                         alpha=0.15, color=color)

    ax1.set_xlabel("Window")
    ax1.set_ylabel("Drift Metric")
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = ax1.twinx()
    acc_arr = np.asarray(accuracy_mean, dtype=float)
    windows = np.arange(len(acc_arr))

    ax2.plot(
        windows,
        acc_arr,
        color="red",
        linewidth=2,
        linestyle="--",
        label="Accuracy (mean)",
    )
    ax2.fill_between(
        windows,
        accuracy_ci_lower,
        accuracy_ci_upper,
        alpha=0.15,
        color="red",
        label="Accuracy 95% CI",
    )

    acc_threshold: float | None = None
    first_drop_idx: int | None = None
    if drift_start > 0:
        pre_drift = acc_arr[:drift_start]
        if len(pre_drift) > 0:
            acc_threshold = float(pre_drift.mean() - 2 * (pre_drift.std() + 1e-10))
            smooth = (
                pd.Series(acc_arr)
                .rolling(3, min_periods=1, center=True)
                .mean()
                .values
            )
            drops = np.where(smooth < acc_threshold)[0]
            if len(drops) > 0:
                first_drop_idx = int(drops[0])

    if acc_threshold is not None:
        ax2.axhline(
            y=acc_threshold,
            color="#c0392b",
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            label="Accuracy drop threshold",
        )

    ax2.set_ylabel("Accuracy", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right", fontsize=8)

    ax1.axvline(x=drift_start, color="gray", linestyle=":", alpha=0.7, label="Drift Start")
    if first_drop_idx is not None:
        ax1.axvline(
            x=first_drop_idx,
            color="#2980b9",
            linestyle="--",
            alpha=0.85,
            label="First accuracy drop",
        )

    ax1.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_alert_timeline(
    alert_levels: list[str],
    accuracy_series: list[float] | np.ndarray,
    drift_start: int,
    title: str,
    save_path: str,
    accuracy_drop_window: int | None = None,
) -> None:
    """Plot alert level bands (OK/WARNING/CRITICAL) with accuracy overlay.

    Args:
        alert_levels: List of alert levels per window ("ok", "warning", "critical").
        accuracy_series: List or array of accuracy values per window.
        drift_start: Window index where drift injection begins.
        title: Title for the plot.
        save_path: File path to save the figure.
        accuracy_drop_window: Optional window index where accuracy drop is detected. Marked on the
            plot if provided.
    """
    _ensure_dir(save_path)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    windows = range(len(alert_levels))
    color_map = {"ok": "#2ecc71", "warning": "#f39c12", "critical": "#e74c3c"}

    # draw alert bands as colored background spans
    for i, level in enumerate(alert_levels):
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.25, color=color_map.get(level, "gray"))

    # accuracy line
    ax1.plot(windows, accuracy_series, color="black", linewidth=2,
             marker="o", markersize=4, label="Accuracy", zorder=3)
    ax1.set_xlabel("Window")
    ax1.set_ylabel("Accuracy")

    ax1.axvline(x=drift_start, color="gray", linestyle=":", alpha=0.7, label="Drift Start")
    if accuracy_drop_window is None:
        acc_arr = np.asarray(accuracy_series, dtype=float)
        pre_drift = acc_arr[:drift_start]
        if len(pre_drift) > 0:
            acc_thresh = pre_drift.mean() - 2 * (pre_drift.std() + 1e-10)
            smooth = (
                pd.Series(acc_arr)
                .rolling(3, min_periods=1, center=True)
                .mean()
                .values
            )
            drops = np.where(smooth < acc_thresh)[0]
            accuracy_drop_window = int(drops[0]) + 1 if len(drops) > 0 else None

    if accuracy_drop_window is not None:
        # convert 1-based window numbering to x-axis index used by the plot.
        drop_x = accuracy_drop_window - 1
        ax1.axvline(
            x=drop_x,
            color="#2980b9",
            linestyle="--",
            alpha=0.8,
            label="Accuracy Drop",
        )

    # legend entries for alert levels
    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.25, label="OK"),
        Patch(facecolor="#f39c12", alpha=0.25, label="WARNING"),
        Patch(facecolor="#e74c3c", alpha=0.25, label="CRITICAL"),
        plt.Line2D([0], [0], color="black", linewidth=2, marker="o", markersize=4,
                   label="Accuracy"),
        plt.Line2D([0], [0], color="gray", linestyle=":", alpha=0.7, label="Drift Start"),
    ]
    if accuracy_drop_window is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color="#2980b9", linestyle="--", alpha=0.8, label="Accuracy Drop")
        )
    ax1.legend(handles=legend_elements, loc="lower left", fontsize=9)
    ax1.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_alert_rate_heatmap(
    alert_rates: dict[str, np.ndarray],
    save_path: str,
) -> None:
    """Heatmap of per-window alert rates across experiments.

    Args:
        alert_rates: Dictionary mapping experiment names to arrays of alert rates per window.
        save_path: File path to save the figure.
    """
    _ensure_dir(save_path)
    names = list(alert_rates.keys())
    data = np.array([alert_rates[n] for n in names])
    n_windows = data.shape[1]

    fig, ax = plt.subplots(figsize=(14, max(3, len(names) * 0.8)))
    sns.heatmap(
        data, ax=ax,
        xticklabels=[str(i + 1) for i in range(n_windows)],
        yticklabels=names,
        cmap="YlOrRd", vmin=0, vmax=1,
        annot=True, fmt=".0%", annot_kws={"fontsize": 8},
        cbar_kws={"label": "Alert Rate (WARNING+)"},
    )
    ax.set_xlabel("Window")
    ax.set_title("Per-Window Alert Rate Across Seeds")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_warning_onset_windows(
    alert_rates: dict[str, np.ndarray],
    drift_start: int,
    save_path: str,
    *,
    majority_threshold: float = 0.5,
) -> None:
    """Plot first warning-onset windows across multi-seed experiment aggregates.

    For each experiment, marks:
    - first window with any WARNING+ activity (alert rate > 0)
    - first window with majority WARNING+ activity (alert rate >= threshold)

    Args:
        alert_rates: Dictionary mapping experiment names to arrays of alert rates per window.
        drift_start: Window index where drift injection begins (marked on the plot).
        save_path: File path to save the figure.
        majority_threshold: Threshold for majority WARNING+ activity (default 0.5 for >50% seeds).
            Must be in (0, 1].
    """
    _ensure_dir(save_path)
    if not alert_rates:
        raise ValueError("alert_rates must not be empty")
    if not (0 < majority_threshold <= 1):
        raise ValueError("majority_threshold must be in (0, 1]")

    names = list(alert_rates.keys())
    data = [np.asarray(alert_rates[n], dtype=float) for n in names]
    n_windows = len(data[0])
    if any(len(arr) != n_windows for arr in data):
        raise ValueError("All alert-rate arrays must have the same length")

    y = np.arange(len(names))
    first_any: list[float] = []
    first_majority: list[float] = []

    for arr in data:
        any_idx = np.where(arr > 0)[0]
        maj_idx = np.where(arr >= majority_threshold)[0]
        first_any.append(float(any_idx[0]) if len(any_idx) > 0 else np.nan)
        first_majority.append(float(maj_idx[0]) if len(maj_idx) > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * len(names))))

    for yi in y:
        ax.hlines(yi, 0, n_windows - 1, color="#dfe6e9", linewidth=1.5, zorder=1)

    ax.scatter(first_any, y, marker="o", s=70, color="#f39c12",
               label="First WARNING+ (rate > 0)", zorder=3)
    ax.scatter(first_majority, y, marker="s", s=70, color="#e74c3c",
               label=f"First majority WARNING+ (rate >= {majority_threshold:.0%})", zorder=4)

    ax.axvline(drift_start, color="gray", linestyle=":", alpha=0.8, label="Drift Start")

    ax.set_xlim(-0.5, n_windows - 0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xticks(np.arange(n_windows))
    ax.set_xticklabels([str(i + 1) for i in range(n_windows)])
    ax.set_xlabel("Window")
    ax.set_title("Warning Onset Windows Across Multi-Seed Aggregates")
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_lead_time_distributions(
    lead_times: dict[str, list[float | None]],
    save_path: str,
) -> None:
    """Box plots of WARNING lead times per experiment.

    Args:
        lead_times: Dictionary mapping experiment names to lists of lead times (in windows) for
                    each seed. Lead times can be float values or None (for no detection).
        save_path: File path to save the figure.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(lead_times.keys())
    box_data = []
    detection_rates = []
    for name in names:
        vals = lead_times[name]
        clean = [v for v in vals if v is not None]
        box_data.append(clean)
        detection_rates.append(len(clean) / len(vals) if vals else 0)

    positions = range(len(names))
    bp = ax.boxplot(
        box_data, positions=list(positions), widths=0.5,
        patch_artist=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.4)

    # strip plot overlay for individual points
    for i, data in enumerate(box_data):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data))
        ax.scatter(
            [i + j for j in jitter], data,
            alpha=0.5, s=20, color="#2c3e50", zorder=3,
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("WARNING Lead Time (windows)")
    ax.set_title("Detection Lead Time Distributions")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    # annotate detection rates
    for i, rate in enumerate(detection_rates):
        ax.annotate(
            f"{rate:.0%} detect",
            xy=(i, ax.get_ylim()[1]),
            ha="center", va="bottom", fontsize=8, color="#7f8c8d",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
