"""End-to-end Transformer drift detection experiment.

Loads a frozen fine-tuned model, generates drifted windows, extracts
attributions (attention or IG), computes drift metrics via the expl_drift
pipeline, and collects results.

Usage:
    PYTHONPATH=/home/brendenadm/projects python experiments/transformer_drift/run_experiment.py \
        --model bert --drift domain_shift --attribution attention --seed 0

    # Or run all combinations:
    PYTHONPATH=/home/brendenadm/projects python experiments/transformer_drift/run_experiment.py --all
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from expl_drift import DriftDetector, DriftMonitor, compute_detection_lead_time
from expl_drift.drift.summarize import summarize_attributions
from expl_drift.explanations.attention import explain_attention
from expl_drift.explanations.ig_transformer import explain_ig_transformer

from experiments.transformer_drift.config import (
    ATTENTION_BATCH_SIZE,
    ATTENTION_STRATEGY,
    ATTRIBUTION_METHODS,
    CRITICAL_STD,
    DRIFT_START_WINDOW,
    DRIFT_TYPES,
    IG_BATCH_SIZE,
    IG_N_STEPS,
    MAX_SEQ_LENGTH,
    MODEL_TYPES,
    MODELS_DIR,
    N_CALIBRATION,
    N_WINDOWS,
    RESULTS_DIR,
    SEEDS,
    WARNING_STD,
)
from experiments.transformer_drift.windowing import create_drifted_windows


# Decoder-only models concentrate attention on the BOS token regardless
# of content ("attention sink").  This dominates drift metrics and masks
# real distributional shifts.  We zero-out the BOS position for these
# architectures after extraction.
_DECODER_MODEL_TYPES = {"gpt2", "phi2", "llama", "mistral"}


def _mask_attention_sink(attrs: np.ndarray, model_type: str) -> np.ndarray:
    """Zero out position 0 (BOS token) for decoder-only models."""
    if model_type in _DECODER_MODEL_TYPES:
        attrs = attrs.copy()
        attrs[:, 0] = 0.0
    return attrs


def load_model(model_type: str):
    """Load fine-tuned model and tokenizer from saved checkpoint."""
    model_dir = MODELS_DIR / f"{model_type}-agnews"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")

    return model, tokenizer


def extract_attributions(
    model,
    tokenizer,
    texts: list[str],
    method: str,
    device: str = "cpu",
    attention_batch_size: int = ATTENTION_BATCH_SIZE,
) -> np.ndarray:
    """Extract attributions using the specified method."""
    if method == "attention":
        return explain_attention(
            model, tokenizer, texts,
            max_length=MAX_SEQ_LENGTH,
            strategy=ATTENTION_STRATEGY,
            batch_size=attention_batch_size,
            device=device,
        )
    elif method == "integrated_gradients":
        return explain_ig_transformer(
            model, tokenizer, texts,
            max_length=MAX_SEQ_LENGTH,
            n_steps=IG_N_STEPS,
            batch_size=IG_BATCH_SIZE,
            device=device,
        )
    else:
        raise ValueError(f"Unknown attribution method: {method}")


def compute_accuracy(model, tokenizer, texts, labels, device="cpu"):
    """Compute classification accuracy on a batch."""
    from expl_drift.explanations.transformer import tokenize_texts

    model.eval()
    all_preds = []

    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenize_texts(tokenizer, batch_texts, max_length=MAX_SEQ_LENGTH, device=device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)

    return np.mean(np.array(all_preds) == np.array(labels))


def run_single_experiment(
    model_type: str,
    drift_type: str,
    attribution_method: str,
    seed: int,
) -> dict:
    """Run one experiment configuration and return results."""
    print(f"\n{'='*60}")
    print(f"Model: {model_type} | Drift: {drift_type} | "
          f"Attribution: {attribution_method} | Seed: {seed}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()

    # Load model
    model, tokenizer = load_model(model_type)

    # Create drifted windows
    windows = create_drifted_windows(
        drift_type=drift_type,
        n_windows=N_WINDOWS,
        start_window=DRIFT_START_WINDOW,
        seed=seed,
    )

    # Extract attributions for all windows
    print("Extracting attributions...")
    all_attributions = []
    accuracies = []
    for wid, window in enumerate(windows):
        texts = window["text"]
        labels = window["label"]

        attrs = extract_attributions(model, tokenizer, texts, attribution_method, device)
        attrs = _mask_attention_sink(attrs, model_type)
        all_attributions.append(attrs)

        acc = compute_accuracy(model, tokenizer, texts, labels, device)
        accuracies.append(acc)
        print(f"  Window {wid:2d}: acc={acc:.3f}, attrs shape={attrs.shape}")

    # Decoder-only models: reduce to position-invariant summary features
    # (positional features are meaningless for causal LMs).
    # Encoder models: use raw positional attributions (positions carry
    # stable signal due to CLS token and bidirectional attention).
    is_decoder = model_type in _DECODER_MODEL_TYPES
    if is_decoder:
        drift_inputs = [summarize_attributions(a) for a in all_attributions]
    else:
        drift_inputs = all_attributions

    # Set up drift detection
    baseline_attrs = drift_inputs[0]
    detector = DriftDetector(baseline_attrs)

    calibration_attrs = drift_inputs[1:1 + N_CALIBRATION]
    monitor = DriftMonitor(
        detector, calibration_attrs,
        warning_std=WARNING_STD,
        critical_std=CRITICAL_STD,
    )

    # Evaluate windows after calibration (don't re-evaluate calibration windows)
    eval_start = 1 + N_CALIBRATION
    alert_levels = []
    metric_results = []
    for wid in range(eval_start, N_WINDOWS):
        result = monitor.evaluate(drift_inputs[wid])
        alert_levels.append(result["alert_level"].value)
        metric_results.append(result["metrics"])

    # Decoder: use max_wasserstein (cosine_drift is near-zero with 6 mixed-scale features)
    # Encoder: use cosine_drift (works well with positional features)
    drift_metric = "max_wasserstein" if is_decoder else "cosine_drift"
    drift_series = np.array([m[drift_metric] for m in metric_results])
    acc_series = np.array(accuracies[eval_start:])
    lead_time = compute_detection_lead_time(drift_series, acc_series)

    elapsed = time.time() - t_start
    print(f"  Lead time: {lead_time} windows | Elapsed: {elapsed:.1f}s")

    return {
        "model_type": model_type,
        "drift_type": drift_type,
        "attribution_method": attribution_method,
        "seed": seed,
        "accuracies": accuracies,
        "alert_levels": alert_levels,
        "metric_results": metric_results,
        "lead_time": lead_time,
        "elapsed_seconds": elapsed,
        "eval_start": eval_start,
    }


def save_results(results: list[dict], output_dir: Path):
    """Save experiment results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    rows = []
    for r in results:
        rows.append({
            "model_type": r["model_type"],
            "drift_type": r["drift_type"],
            "attribution_method": r["attribution_method"],
            "seed": r["seed"],
            "lead_time": r["lead_time"],
            "final_accuracy": r["accuracies"][-1],
            "baseline_accuracy": r["accuracies"][0],
            "n_warnings": sum(1 for a in r["alert_levels"] if a == "warning"),
            "n_criticals": sum(1 for a in r["alert_levels"] if a == "critical"),
            "first_warning_window": next(
                (i + r["eval_start"] for i, a in enumerate(r["alert_levels"]) if a != "ok"), None
            ),
            "elapsed_seconds": r["elapsed_seconds"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\nSummary saved to {output_dir / 'summary.csv'}")

    # Full per-window results
    per_window_rows = []
    for r in results:
        eval_start = r["eval_start"]
        for wid in range(len(r["accuracies"])):
            row = {
                "model_type": r["model_type"],
                "drift_type": r["drift_type"],
                "attribution_method": r["attribution_method"],
                "seed": r["seed"],
                "window_id": wid,
                "accuracy": r["accuracies"][wid],
            }
            if wid >= eval_start:
                eval_idx = wid - eval_start
                row["alert_level"] = r["alert_levels"][eval_idx]
                row.update(r["metric_results"][eval_idx])
            per_window_rows.append(row)
    df_windows = pd.DataFrame(per_window_rows)
    df_windows.to_csv(output_dir / "per_window_results.csv", index=False)
    print(f"Per-window results saved to {output_dir / 'per_window_results.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Transformer drift detection experiment")
    parser.add_argument("--model", choices=MODEL_TYPES, help="Model type")
    parser.add_argument("--drift", choices=DRIFT_TYPES, help="Drift type")
    parser.add_argument("--attribution", choices=ATTRIBUTION_METHODS, help="Attribution method")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--all", action="store_true", help="Run all combinations")
    args = parser.parse_args()

    if args.all:
        configs = [
            (m, d, a, s)
            for m in MODEL_TYPES
            for d in DRIFT_TYPES
            for a in ATTRIBUTION_METHODS
            for s in SEEDS
        ]
    else:
        if not all([args.model, args.drift, args.attribution, args.seed is not None]):
            parser.error("Specify --model, --drift, --attribution, --seed or use --all")
        configs = [(args.model, args.drift, args.attribution, args.seed)]

    results = []
    for model_type, drift_type, attr_method, seed in configs:
        result = run_single_experiment(model_type, drift_type, attr_method, seed)
        results.append(result)

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"run_{timestamp}"
    save_results(results, output_dir)

    # Print summary
    if len(results) > 1:
        df = pd.read_csv(output_dir / "summary.csv")
        print("\n=== Aggregate Summary ===")
        print(df.groupby(["model_type", "drift_type", "attribution_method"]).agg({
            "lead_time": ["mean", "std"],
            "final_accuracy": "mean",
            "n_warnings": "mean",
            "first_warning_window": "mean",
        }).round(2))


if __name__ == "__main__":
    main()
