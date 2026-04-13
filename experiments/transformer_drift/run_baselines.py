"""Baseline drift detection for comparison against attribution drift.

Runs the same windowing and detection pipeline but swaps attribution
features for simpler signals that any monitoring system could use:

  - **confidence**:   per-sample softmax outputs ``(n, n_classes)``.
                      Catches shifts the model is uncertain about.
  - **input_stats**:  per-sample text statistics ``(n, 3)`` — token
                      count, unique-token count, mean word length.
                      Catches shifts in raw input properties.

If attribution drift fires when these baselines don't, the paper has
a real contribution.  If these catch everything too, the attribution
approach isn't adding value.

Usage:
    PYTHONPATH=/home/brendenadm/projects \
        python experiments/transformer_drift/run_baselines.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd
import torch

from expl_drift import DriftDetector, DriftMonitor, compute_detection_lead_time
from expl_drift.explanations.transformer import tokenize_texts

from experiments.transformer_drift.config import (
    CRITICAL_STD,
    DRIFT_TYPES,
    MAX_SEQ_LENGTH,
    N_CALIBRATION,
    N_WINDOWS,
    RESULTS_DIR,
    SEEDS,
    WARNING_STD,
)
from experiments.transformer_drift.run_experiment import load_model, compute_accuracy, save_results
from experiments.transformer_drift.windowing import create_drifted_windows


def extract_confidence(model, tokenizer, texts: list[str], device: str) -> np.ndarray:
    """Return per-sample softmax outputs ``(n_samples, n_classes)``."""
    model.eval()
    all_probs = []
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenize_texts(tokenizer, batch_texts, max_length=MAX_SEQ_LENGTH, device=device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def extract_input_stats(texts: list[str]) -> np.ndarray:
    """Return per-sample text statistics ``(n_samples, 3)``: token_count, unique_tokens, mean_word_len."""
    stats = np.zeros((len(texts), 3), dtype=float)
    for i, text in enumerate(texts):
        words = text.split()
        n_words = len(words)
        if n_words == 0:
            continue
        stats[i, 0] = n_words
        stats[i, 1] = len(set(words))
        stats[i, 2] = float(np.mean([len(w) for w in words]))
    return stats


def run_baseline_experiment(
    model_type: str,
    drift_type: str,
    baseline_method: str,
    seed: int,
) -> dict:
    """Run one baseline configuration."""
    print(f"\n{'='*60}")
    print(f"Model: {model_type} | Drift: {drift_type} | "
          f"Baseline: {baseline_method} | Seed: {seed}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()

    model, tokenizer = load_model(model_type)

    windows = create_drifted_windows(
        drift_type=drift_type,
        n_windows=N_WINDOWS,
        start_window=5,
        seed=seed,
    )

    all_features = []
    accuracies = []
    for wid, window in enumerate(windows):
        texts = window["text"]
        labels = window["label"]

        if baseline_method == "confidence":
            features = extract_confidence(model, tokenizer, texts, device)
        elif baseline_method == "input_stats":
            features = extract_input_stats(texts)
        else:
            raise ValueError(f"Unknown baseline method: {baseline_method}")

        all_features.append(features)
        acc = compute_accuracy(model, tokenizer, texts, labels, device)
        accuracies.append(acc)
        print(f"  Window {wid:2d}: acc={acc:.3f}, features shape={features.shape}")

    baseline_features = all_features[0]
    detector = DriftDetector(baseline_features)
    calibration = all_features[1:1 + N_CALIBRATION]
    monitor = DriftMonitor(detector, calibration, warning_std=WARNING_STD, critical_std=CRITICAL_STD)

    eval_start = 1 + N_CALIBRATION
    alert_levels = []
    metric_results = []
    for wid in range(eval_start, N_WINDOWS):
        result = monitor.evaluate(all_features[wid])
        alert_levels.append(result["alert_level"].value)
        metric_results.append(result["metrics"])

    # Use the same metric selection as attribution pipeline:
    # low-dim feature spaces → max_wasserstein, else cosine_drift.
    n_features = baseline_features.shape[1]
    drift_metric = "max_wasserstein" if n_features <= 10 else "cosine_drift"
    drift_series = np.array([m[drift_metric] for m in metric_results])
    acc_series = np.array(accuracies[eval_start:])
    lead_time = compute_detection_lead_time(drift_series, acc_series)

    elapsed = time.time() - t_start
    print(f"  Lead time: {lead_time} | Elapsed: {elapsed:.1f}s")

    del model, tokenizer, all_features, detector, monitor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_type": model_type,
        "drift_type": drift_type,
        "attribution_method": baseline_method,  # reuse column name for save_results compat
        "seed": seed,
        "accuracies": accuracies,
        "alert_levels": alert_levels,
        "metric_results": metric_results,
        "lead_time": lead_time,
        "elapsed_seconds": elapsed,
        "eval_start": eval_start,
    }


def main():
    # Only run on models we have; skip IG-style combinations
    baseline_methods = ["confidence", "input_stats"]
    configs = [
        (m, d, b, s)
        for m in ["bert"]
        for d in DRIFT_TYPES
        for b in baseline_methods
        for s in SEEDS
    ]

    print(f"Running {len(configs)} baseline configs")
    results = []
    for model_type, drift_type, method, seed in configs:
        result = run_baseline_experiment(model_type, drift_type, method, seed)
        results.append(result)

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"run_baselines_{ts}"
    save_results(results, output_dir)
    print(f"Done. Results in {output_dir}")


if __name__ == "__main__":
    main()
