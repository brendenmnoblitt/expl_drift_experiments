"""Baseline metrics for model performance and raw data drift."""

import numpy as np
import pandas as pd
from expl_drift.drift.metrics import (
    compute_energy_distance,
    compute_jsd,
    compute_ks_statistic,
    compute_max_jsd,
)

from expl_drift_experiments.model.predictor import evaluate_window


def compute_accuracy_per_window(
    model: object,
    windows: list[tuple[np.ndarray, np.ndarray]],
    model_type: str = "xgboost",
) -> list[float]:
    """Compute accuracy for each window. Returns list of accuracy scores.
    
    Args:
        model: Trained model.
        windows: List of (X_window, y_window) tuples.
        model_type: Type of model (e.g. "xgboost", "sklearn"). Passed to evaluate_window.
    Returns:
        list[float]: Accuracy for each window.
    """
    accuracies: list[float] = []
    for X_w, y_w in windows:
        metrics = evaluate_window(model, X_w, y_w, model_type)
        accuracies.append(metrics["accuracy"])
    return accuracies


def compute_data_drift_per_window(
    baseline_X: np.ndarray,
    windows: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Compute drift metrics on raw feature distributions (not attributions).

    Args:
        baseline_X: Feature data for the baseline window (window 0).
        windows: List of (X_window, y_window) tuples for all windows.
    Returns:
        pd.DataFrame: DataFrame with drift metrics for each window, indexed by window number.
    """
    base_arr = baseline_X.values if hasattr(baseline_X, "values") else np.asarray(baseline_X)
    records: list[dict[str, float | int]] = []
    for i, (X_w, _) in enumerate(windows):
        curr_arr = X_w.values if hasattr(X_w, "values") else np.asarray(X_w)
        jsd = compute_jsd(base_arr, curr_arr)
        max_jsd = compute_max_jsd(base_arr, curr_arr)
        energy = compute_energy_distance(base_arr, curr_arr)
        ks_max, ks_frac = compute_ks_statistic(base_arr, curr_arr)
        records.append({
            "window": i,
            "data_jsd": jsd,            # mean per-feature JSD
            "data_max_jsd": max_jsd,    # max per-feature JSD — similar to explanation max_jsd
            "data_energy": energy,
            "data_ks_max": ks_max,      # max per-feature KS statistic
            "data_ks_frac_sig": ks_frac,
        })
    return pd.DataFrame(records).set_index("window")
