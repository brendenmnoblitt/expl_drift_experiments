"""Multi-seed INSECTS natural drift experiment.

Runs the explanation drift monitoring pipeline on the INSECTS abrupt_balanced
dataset (Souza et al. 2020) across 25 seeds. The dataset has known abrupt
drift points induced by temperature changes, providing a natural drift
benchmark with discrete regime transitions.

Usage:
    PYTHONPATH=/home/brendenadm/projects python scripts/insects_experiment.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats as sp_stats

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from expl_drift import DriftDetector, DriftMonitor, explain_shap
from expl_drift_experiments import (
    load_insects_dataset,
    partition_chronological,
    train_xgboost,
    evaluate_window,
    plot_multi_seed_drift,
)

# === Configuration ===
SEEDS = list(range(25))
N_WINDOWS = 20
N_CALIBRATION = 4
WARNING_STD = 2.5
CRITICAL_STD = 3.5

# Known drift points for abrupt_balanced (sample indices)
DRIFT_POINTS = [14352, 19500, 33240, 38682, 39510]

# Output
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "insects"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_insects_single_seed(seed: int) -> dict:
    """Run natural drift experiment on INSECTS for one seed."""
    X, y = load_insects_dataset()
    windows = partition_chronological(X, y, n_windows=N_WINDOWS)
    X_base, y_base = windows[0]

    xgb_model, _ = train_xgboost(X_base, y_base, seed=seed)
    shap_baseline = explain_shap(xgb_model, X_base, X_base, model_type="xgboost")
    detector = DriftDetector(shap_baseline)

    # Compute SHAP for all windows
    all_shap = [shap_baseline]
    for wid in range(1, N_WINDOWS):
        X_w, _ = windows[wid]
        all_shap.append(explain_shap(xgb_model, X_w, X_base, model_type="xgboost"))

    calibration_shap = all_shap[1 : 1 + N_CALIBRATION]
    monitor = DriftMonitor(
        detector,
        calibration_shap,
        warning_std=WARNING_STD,
        critical_std=CRITICAL_STD,
    )

    accuracies, alert_levels = [], []
    cosine_drift, max_jsd, max_wasserstein = [], [], []

    for wid in range(1, N_WINDOWS):
        X_w, y_w = windows[wid]
        acc = evaluate_window(xgb_model, X_w, y_w)["accuracy"]
        result = monitor.evaluate(all_shap[wid])

        accuracies.append(acc)
        alert_levels.append(result["alert_level"].value)
        cosine_drift.append(result["metrics"]["cosine_drift"])
        max_jsd.append(result["metrics"]["max_jsd"])
        max_wasserstein.append(result["metrics"]["max_wasserstein"])

    return {
        "accuracies": np.array(accuracies),
        "alert_levels": alert_levels,
        "cosine_drift": np.array(cosine_drift),
        "max_jsd": np.array(max_jsd),
        "max_wasserstein": np.array(max_wasserstein),
    }


def compute_lead_times(result, drift_start=5):
    """Compute lead times relative to first known drift point."""
    alerts = result["alert_levels"]
    acc = result["accuracies"]

    warn_indices = [i for i, a in enumerate(alerts) if a == "warning"]
    crit_indices = [i for i, a in enumerate(alerts) if a == "critical"]
    first_warn = (warn_indices[0] + 1) if warn_indices else None
    first_crit = (crit_indices[0] + 1) if crit_indices else None

    pre_drift = acc[: drift_start - 1]
    acc_thresh = pre_drift.mean() - 2 * (pre_drift.std() + 1e-10)
    smooth = pd.Series(acc).rolling(3, min_periods=1, center=True).mean().values
    drops = np.where(smooth < acc_thresh)[0]
    first_drop = int(drops[0]) + 1 if len(drops) > 0 else None

    warn_lead = (first_drop - first_warn) if (first_drop and first_warn) else None
    crit_lead = (first_drop - first_crit) if (first_drop and first_crit) else None

    alert_detected = any(a in ("warning", "critical") for a in alerts)

    return {
        "first_warning": first_warn,
        "first_critical": first_crit,
        "accuracy_drop": first_drop,
        "warning_lead": warn_lead,
        "critical_lead": crit_lead,
        "alert_detected": alert_detected,
    }


def bootstrap_ci(values, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for median."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return np.median(clean) if clean else None, None, None
    rng = np.random.default_rng(42)
    boots = np.array([np.median(rng.choice(clean, len(clean))) for _ in range(n_boot)])
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return np.median(clean), lo, hi


def main():
    # Compute which windows contain drift points
    X, y = load_insects_dataset()
    window_size = len(X) // N_WINDOWS
    drift_windows = sorted(set(p // window_size for p in DRIFT_POINTS))
    print(f"INSECTS abrupt_balanced: {len(X)} samples, {N_WINDOWS} windows of ~{window_size}")
    print(f"Known drift points at windows: {drift_windows}")
    print(f"Running {len(SEEDS)} seeds...\n")

    # Run all seeds
    all_results = []
    for i, seed in enumerate(SEEDS):
        if i % 5 == 0:
            print(f"  Seed {i + 1}/{len(SEEDS)}...", flush=True)
        result = run_insects_single_seed(seed)
        all_results.append(result)

    # Aggregate
    n_seeds = len(all_results)
    n_win = len(all_results[0]["accuracies"])
    t_crit = sp_stats.t.ppf(0.975, df=n_seeds - 1)

    acc_stack = np.array([r["accuracies"] for r in all_results])
    cos_stack = np.array([r["cosine_drift"] for r in all_results])
    jsd_stack = np.array([r["max_jsd"] for r in all_results])
    was_stack = np.array([r["max_wasserstein"] for r in all_results])

    def mean_ci(stack):
        m = stack.mean(axis=0)
        sem = stack.std(axis=0, ddof=1) / np.sqrt(n_seeds)
        return m, m - t_crit * sem, m + t_crit * sem

    acc_mean, acc_ci_lo, acc_ci_hi = mean_ci(acc_stack)
    cos_mean, cos_ci_lo, cos_ci_hi = mean_ci(cos_stack)
    jsd_mean, jsd_ci_lo, jsd_ci_hi = mean_ci(jsd_stack)
    was_mean, was_ci_lo, was_ci_hi = mean_ci(was_stack)

    windows = list(range(n_win))
    mean_df = pd.DataFrame(
        {"cosine_drift": cos_mean, "max_jsd": jsd_mean, "max_wasserstein": was_mean},
        index=windows,
    )
    ci_lo_df = pd.DataFrame(
        {"cosine_drift": cos_ci_lo, "max_jsd": jsd_ci_lo, "max_wasserstein": was_ci_lo},
        index=windows,
    )
    ci_hi_df = pd.DataFrame(
        {"cosine_drift": cos_ci_hi, "max_jsd": jsd_ci_hi, "max_wasserstein": was_ci_hi},
        index=windows,
    )

    # Alert rate per window
    alert_rate = np.zeros(n_win)
    for w in range(n_win):
        n_alerting = sum(
            1 for r in all_results if r["alert_levels"][w] in ("warning", "critical")
        )
        alert_rate[w] = n_alerting / n_seeds

    # Lead times
    # First drift point is at window 5, so use drift_start=5
    first_drift_window = drift_windows[0]
    lead_times = [compute_lead_times(r, drift_start=first_drift_window) for r in all_results]
    warning_leads = [lt["warning_lead"] for lt in lead_times]
    clean_leads = [v for v in warning_leads if v is not None]
    alert_detection_rate = sum(1 for lt in lead_times if lt["alert_detected"]) / n_seeds
    lead_time_det_rate = len(clean_leads) / n_seeds

    # Spearman
    all_was = np.concatenate([r["max_wasserstein"] for r in all_results])
    all_acc = np.concatenate([r["accuracies"] for r in all_results])
    rho, pval = spearmanr(all_was, all_acc)

    median_lead, ci_lo_lead, ci_hi_lead = bootstrap_ci(warning_leads)

    # Print summary
    print("\n=== INSECTS Abrupt Balanced Results ===")
    print(f"Alert Detection Rate:      {alert_detection_rate:.0%}")
    print(f"Lead-Time Detection Rate:  {lead_time_det_rate:.0%}")
    if median_lead is not None and ci_lo_lead is not None:
        print(f"Median Lead [95% CI]:      {median_lead:.1f} [{ci_lo_lead:.1f}, {ci_hi_lead:.1f}]")
    else:
        print("Median Lead [95% CI]:      N/A")
    if clean_leads:
        mean_lead = np.mean(clean_leads)
        sem_lead = np.std(clean_leads, ddof=1) / np.sqrt(len(clean_leads))
        print(f"Mean Lead ± SEM:           {mean_lead:.1f} ± {sem_lead:.1f}")
    else:
        print("Mean Lead ± SEM:           N/A")
    print(f"Spearman rho:              {rho:.3f} (p={pval:.4f})")

    print("\nPer-window accuracy (mean across seeds):")
    for w in range(n_win):
        drift_marker = " <-- DRIFT" if (w + 1) in drift_windows or w in drift_windows else ""
        alert_pct = f"{alert_rate[w]:.0%}"
        print(
            f"  Window {w + 1:2d}: acc={acc_mean[w]:.4f}  "
            f"wasserstein={was_mean[w]:.4f}  alert_rate={alert_pct}{drift_marker}"
        )

    # Save plot
    plot_multi_seed_drift(
        mean_df,
        ci_lo_df,
        ci_hi_df,
        acc_mean,
        acc_ci_lo,
        acc_ci_hi,
        drift_start=first_drift_window - 1,
        title=f"INSECTS Abrupt Drift: Mean ± 95% CI ({n_seeds} seeds)",
        save_path=str(RESULTS_DIR / "insects_abrupt_aggregate.png"),
    )
    print(f"\nFigure saved to {RESULTS_DIR / 'insects_abrupt_aggregate.png'}")

    # Save summary CSV
    summary = {
        "Experiment": "Natural (INSECTS)",
        "Alert Detection Rate": f"{alert_detection_rate:.0%}",
        "Lead-Time Detection Rate": f"{lead_time_det_rate:.0%}",
        "Median Lead [95% CI]": (
            f"{median_lead:.1f} [{ci_lo_lead:.1f}, {ci_hi_lead:.1f}]"
            if median_lead is not None and ci_lo_lead is not None
            else "N/A"
        ),
        "Mean Lead ± SEM": (
            f"{np.mean(clean_leads):.1f} ± {np.std(clean_leads, ddof=1) / np.sqrt(len(clean_leads)):.1f}"
            if len(clean_leads) > 1
            else "N/A"
        ),
        "Spearman rho": f"{rho:.3f}",
        "Spearman p": f"{pval:.4f}",
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "insects_summary.csv", index=False)
    print(f"Summary saved to {RESULTS_DIR / 'insects_summary.csv'}")


if __name__ == "__main__":
    main()
