"""Baseline drift detection for comparison against attribution drift.

Runs the same windowing and detection pipeline but swaps attribution
features for simpler signals that any monitoring system could use:

  - **confidence**:    per-sample softmax outputs ``(n, n_classes)``.
                       Catches shifts the model is uncertain about.
  - **input_stats**:   per-sample text statistics ``(n, 3)`` — token
                       count, unique-token count, mean word length.
                       Catches shifts in raw input properties.
  - **cls_embedding**: per-sample pooled hidden state ``(n, hidden_dim)``.
                       For encoders: last-layer [CLS] token. For decoders:
                       last non-padding token.  This is the strong baseline
                       — it's the model's own summary of each input and
                       matches what Alibi Detect / Evidently use in practice.

If attribution drift fires when these baselines don't, the paper has
a real contribution.  If these catch everything too, the attribution
approach isn't adding value.

Usage:
    PYTHONPATH=/home/brendenadm/projects \
        python experiments/transformer_drift/run_baselines.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd
import torch

from expl_drift import DriftDetector, DriftMonitor, compute_detection_lead_time
from expl_drift.drift.metrics import MONITORED_METRICS
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
from experiments.transformer_drift.run_experiment import (
    _DECODER_MODEL_TYPES,
    _pick_device,
    compute_accuracy,
    load_model,
    save_results,
)
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


def extract_cls_embedding(
    model, tokenizer, texts: list[str], device: str, model_type: str
) -> np.ndarray:
    """Return per-sample pooled hidden-state embedding ``(n_samples, hidden_dim)``.

    For encoders, uses the last-layer [CLS] token (position 0). For decoders,
    uses the last non-padding token, matching how
    ``AutoModelForSequenceClassification`` pools for the classification head.
    This is the standard "embedding drift" signal used by drift-detection
    libraries (Alibi Detect, Evidently) and is the strong baseline attribution
    drift must beat or complement.
    """
    model.eval()
    is_decoder = model_type in _DECODER_MODEL_TYPES
    all_embeds = []
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenize_texts(tokenizer, batch_texts, max_length=MAX_SEQ_LENGTH, device=device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)
        if is_decoder:
            # last non-padding token per sample
            lengths = inputs["attention_mask"].sum(dim=1) - 1
            idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
            pooled = hidden.gather(1, idx).squeeze(1)
        else:
            pooled = hidden[:, 0, :]
        all_embeds.append(pooled.cpu().float().numpy())
    return np.concatenate(all_embeds, axis=0)


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

    device = _pick_device()
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
        elif baseline_method == "cls_embedding":
            features = extract_cls_embedding(model, tokenizer, texts, device, model_type)
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

    # Max-of-metrics ensemble: z-score each monitored metric against its
    # calibration baseline, then take the per-window max.  Matches the
    # lead-time computation used for attribution drift in run_experiment.py.
    z_arrays = []
    for metric in MONITORED_METRICS:
        vals = np.array([m[metric] for m in metric_results])
        t = monitor.thresholds[metric]
        z_arrays.append((vals - t["mean"]) / (t["std"] + 1e-10))
    drift_series = np.max(np.stack(z_arrays), axis=0)

    acc_series = np.array(accuracies[eval_start:])
    pre_drift_acc = np.array(accuracies[:eval_start], dtype=float)
    lead_time = compute_detection_lead_time(
        drift_series,
        acc_series,
        drift_baseline=(0.0, 1.0),  # z-scored: baseline (0, 1) by construction
        accuracy_baseline=(float(pre_drift_acc.mean()), float(pre_drift_acc.std())),
    )

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
    all_methods = ["confidence", "input_stats", "cls_embedding"]
    all_models = ["bert", "gpt2"]

    parser = argparse.ArgumentParser(description="Baseline drift detection runs")
    parser.add_argument("--method", choices=all_methods, help="Single baseline method to run")
    parser.add_argument("--model", choices=all_models, help="Single model to run")
    parser.add_argument("--drift", choices=DRIFT_TYPES, help="Single drift type to run")
    parser.add_argument("--seeds", type=int, default=None, help="Override seed count (0..N-1)")
    args = parser.parse_args()

    methods = [args.method] if args.method else all_methods
    models = [args.model] if args.model else all_models
    drifts = [args.drift] if args.drift else DRIFT_TYPES
    seeds = list(range(args.seeds)) if args.seeds is not None else SEEDS

    configs = [
        (m, d, b, s)
        for m in models
        for d in drifts
        for b in methods
        for s in seeds
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
