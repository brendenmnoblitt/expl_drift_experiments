"""Evaluation baselines and plotting utilities for experiments."""

from .baselines import compute_accuracy_per_window, compute_data_drift_per_window
from .visualizer import (
    plot_detection_comparison,
    plot_drift_timeseries,
    plot_explanation_vs_data_drift,
    plot_feature_attribution_heatmap,
    plot_method_consistency,
    plot_multi_seed_drift,
    plot_severity_sweep,
)

__all__ = [
    "compute_accuracy_per_window",
    "compute_data_drift_per_window",
    "plot_drift_timeseries",
    "plot_explanation_vs_data_drift",
    "plot_feature_attribution_heatmap",
    "plot_method_consistency",
    "plot_detection_comparison",
    "plot_severity_sweep",
    "plot_multi_seed_drift",
]
