"""Experiment package for reproducing paper analyses."""

from .data.loader import (
    load_dataset,
    load_credit_dataset,
    load_electricity_dataset,
    partition_into_windows,
    partition_chronological,
    get_baseline_window,
    get_window,
)
from .data.drift_injector import (
    inject_covariate_drift,
    inject_concept_drift,
    inject_concept_drift_conditional,
    inject_concept_drift_perturbation,
    inject_concept_drift_rotation,
    inject_mixed_drift,
)
from .model.trainer import train_xgboost, train_nn
from .model.predictor import predict_batch, evaluate_window
# Explainers live in the core library; re-export for convenience
from expl_drift.explanations import explain_shap, explain_lime, explain_ig
from .evaluation.baselines import compute_accuracy_per_window, compute_data_drift_per_window
from .evaluation.visualizer import (
    plot_drift_timeseries,
    plot_explanation_vs_data_drift,
    plot_feature_attribution_heatmap,
    plot_method_consistency,
    plot_detection_comparison,
    plot_severity_sweep,
    plot_multi_seed_drift,
    plot_alert_timeline,
    plot_alert_rate_heatmap,
    plot_warning_onset_windows,
    plot_lead_time_distributions,
)

__all__ = [
    "load_dataset",
    "load_credit_dataset",
    "load_electricity_dataset",
    "partition_into_windows",
    "partition_chronological",
    "get_baseline_window",
    "get_window",
    "inject_covariate_drift",
    "inject_concept_drift",
    "inject_concept_drift_conditional",
    "inject_concept_drift_perturbation",
    "inject_concept_drift_rotation",
    "inject_mixed_drift",
    "train_xgboost",
    "train_nn",
    "predict_batch",
    "evaluate_window",
    "explain_shap",
    "explain_lime",
    "explain_ig",
    "compute_accuracy_per_window",
    "compute_data_drift_per_window",
    "plot_drift_timeseries",
    "plot_explanation_vs_data_drift",
    "plot_feature_attribution_heatmap",
    "plot_method_consistency",
    "plot_detection_comparison",
    "plot_severity_sweep",
    "plot_multi_seed_drift",
    "plot_alert_timeline",
    "plot_alert_rate_heatmap",
    "plot_warning_onset_windows",
    "plot_lead_time_distributions",
]
