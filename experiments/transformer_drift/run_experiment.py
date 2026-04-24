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
from expl_drift.drift.metrics import MONITORED_METRICS
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

_USE_COLOR = sys.stdout.isatty()
_RESET = "\033[0m" if _USE_COLOR else ""
_DIM = "\033[2m" if _USE_COLOR else ""
_CYAN = "\033[36m" if _USE_COLOR else ""
_GREEN = "\033[32m" if _USE_COLOR else ""
_YELLOW = "\033[33m" if _USE_COLOR else ""
_RED = "\033[31m" if _USE_COLOR else ""
_GRAY = "\033[90m" if _USE_COLOR else ""
_ALERT_COLORS = {
    "ok": _GREEN,
    "warning": _YELLOW,
    "critical": _RED,
    "prewarning": _CYAN,
}


def _summary_bar(
    accuracies: list[float],
    alert_levels: list[str],
    eval_start: int,
    drift_start: int,
    acc_threshold: float,
    drift_flag_window: int | None,
    acc_flag_window: int | None,
    lead_time: int | None,
) -> str:
    """Compact end-of-seed visualization: one colored block per window.

    Shows both flag windows used by ``compute_detection_lead_time`` so the
    lead time is visually measurable as the gap between ★ and ◆.
    """
    glyphs = []
    for wid, acc in enumerate(accuracies):
        if wid < eval_start:
            glyphs.append(f"{_GRAY}·{_RESET}")
            continue
        alert = alert_levels[wid - eval_start]
        ch = "●"
        if alert == "critical":
            glyphs.append(f"{_RED}{ch}{_RESET}")
        elif alert == "warning":
            glyphs.append(f"{_YELLOW}{ch}{_RESET}")
        elif alert == "prewarning":
            glyphs.append(f"{_CYAN}{ch}{_RESET}")
        else:
            glyphs.append(f"{_GREEN}{ch}{_RESET}")

    drift_flag_abs = (eval_start + drift_flag_window) if drift_flag_window is not None else None
    acc_flag_abs = (eval_start + acc_flag_window) if acc_flag_window is not None else None

    drift_markers = []
    acc_markers = []
    for wid, acc in enumerate(accuracies):
        drift_markers.append(f"{_GREEN}★{_RESET}" if wid == drift_flag_abs else " ")
        if wid == acc_flag_abs:
            acc_markers.append(f"{_RED}◆{_RESET}")
        elif acc < acc_threshold and wid >= eval_start:
            acc_markers.append(f"{_DIM}▼{_RESET}")
        else:
            acc_markers.append(" ")

    # insert drift-start separator
    sep = f"{_CYAN}|{_RESET}"
    glyphs.insert(drift_start, sep)
    acc_markers.insert(drift_start, " ")
    drift_markers.insert(drift_start, " ")

    header = "        " + " ".join(
        f"{i:2d}" if i != drift_start else "  " for i in range(len(accuracies) + 1)
    )
    g_line = "  win:  " + "  ".join(glyphs)
    d_line = "  drift:" + "  ".join(drift_markers)
    a_line = "  acc↓: " + "  ".join(acc_markers)

    lead_str = (
        f"lead_time={lead_time:+d}"
        if lead_time is not None
        else "lead_time=None"
    )
    legend = (
        f"  legend: {_GRAY}·{_RESET}=pre-drift  {_GREEN}●{_RESET}=ok  "
        f"{_YELLOW}●{_RESET}=warning  {_RED}●{_RESET}=critical  "
        f"{_CYAN}|{_RESET}=drift starts  {_GREEN}★{_RESET}=drift flag  "
        f"{_RED}◆{_RESET}=acc flag  {_DIM}▼{_RESET}=acc<threshold  "
        f"({lead_str} = ◆ − ★)"
    )
    return "\n".join([header, g_line, d_line, a_line, legend])


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

    device = _pick_device()
    if device != "cpu":
        model = model.to(device)

    return model, tokenizer


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

    device = _pick_device()
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

    is_decoder = model_type in _DECODER_MODEL_TYPES
    drift_metric = "max_wasserstein" if is_decoder else "cosine_drift"
    eval_start = 1 + N_CALIBRATION

    # Extract attributions and evaluate monitor inline so alerts show live.
    print("Extracting attributions...")
    drift_inputs: list = []
    accuracies: list[float] = []
    alert_levels: list[str] = []
    metric_results: list[dict] = []
    calibration_attrs: list = []
    detector: DriftDetector | None = None
    monitor: DriftMonitor | None = None

    for wid, window in enumerate(windows):
        texts = window["text"]
        labels = window["label"]

        attrs = extract_attributions(model, tokenizer, texts, attribution_method, device)
        attrs = _mask_attention_sink(attrs, model_type)
        drift_input = summarize_attributions(attrs) if is_decoder else attrs
        drift_inputs.append(drift_input)

        acc = compute_accuracy(model, tokenizer, texts, labels, device)
        accuracies.append(acc)

        drift_marker = f" {_CYAN}← DRIFT INJECTION BEGINS{_RESET}" if wid == DRIFT_START_WINDOW else ""

        if wid == 0:
            print(f"  Window {wid:2d}: acc={acc:.3f}  {_DIM}[baseline]{_RESET}{drift_marker}")
        elif wid < eval_start:
            calibration_attrs.append(drift_input)
            print(f"  Window {wid:2d}: acc={acc:.3f}  {_DIM}[calibration]{_RESET}{drift_marker}")
        else:
            if monitor is None:
                detector = DriftDetector(drift_inputs[0])
                monitor = DriftMonitor(
                    detector, calibration_attrs,
                    warning_std=WARNING_STD,
                    critical_std=CRITICAL_STD,
                )
            result = monitor.evaluate(drift_input)
            alert = result["alert_level"].value
            alert_levels.append(alert)
            metric_results.append(result["metrics"])
            color = _ALERT_COLORS.get(alert, "")
            # Live ensemble z-score: max over all monitored metrics.
            z_vals = [
                (result["metrics"][m] - monitor.thresholds[m]["mean"])
                / (monitor.thresholds[m]["std"] + 1e-10)
                for m in MONITORED_METRICS
            ]
            z_max = max(z_vals)
            z_argmax = MONITORED_METRICS[int(np.argmax(z_vals))]
            print(
                f"  Window {wid:2d}: acc={acc:.3f}  "
                f"[{color}{alert:>8}{_RESET}] z_max={z_max:+.2f} ({z_argmax}){drift_marker}"
            )

    assert monitor is not None  # guaranteed since eval_start < N_WINDOWS

    # Max-of-metrics ensemble for lead-time.  Z-score each monitored metric
    # against its calibration baseline, then take the per-window max.  Different
    # metrics are sensitive to different distributional shifts (e.g. cosine
    # misses label-noise drift; JSD / Wasserstein pick it up).  Max-of-z
    # catches whichever metric fires first, matching the approach in the
    # original ML-paper framework.
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
        drift_baseline=(0.0, 1.0),  # z-scored: baseline is (0, 1) by construction
        accuracy_baseline=(float(pre_drift_acc.mean()), float(pre_drift_acc.std())),
    )

    elapsed = time.time() - t_start

    # Compact visualization: both flag windows used by compute_detection_lead_time.
    # lead_time = acc_flag_window - drift_flag_window (both indexed from eval_start).
    acc_threshold = float(pre_drift_acc.mean()) - 2.0 * float(pre_drift_acc.std())
    # z-scored ensemble threshold is simply 2.0 (baseline mean=0, std=1 by construction)
    drift_flag_window = next(
        (i for i, v in enumerate(drift_series) if v > 2.0),
        None,
    )
    acc_flag_window = next(
        (i for i, a in enumerate(acc_series) if a < acc_threshold),
        None,
    )
    print()
    print(_summary_bar(
        accuracies, alert_levels, eval_start,
        drift_start=DRIFT_START_WINDOW,
        acc_threshold=acc_threshold,
        drift_flag_window=drift_flag_window,
        acc_flag_window=acc_flag_window,
        lead_time=lead_time,
    ))
    print(f"\n  Lead time: {lead_time} windows | Elapsed: {elapsed:.1f}s")

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


def _json_default(obj):
    """Coerce numpy types for json.dumps."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not JSON-serializable: {type(obj).__name__}")


def _seed_file(output_dir: Path, model: str, drift: str, method: str, seed: int) -> Path:
    return output_dir / f"{model}__{drift}__{method}__seed{seed}.json"


def main():
    parser = argparse.ArgumentParser(description="Transformer drift detection experiment")
    parser.add_argument("--model", choices=MODEL_TYPES, help="Model type")
    parser.add_argument("--drift", choices=DRIFT_TYPES, help="Drift type")
    parser.add_argument("--attribution", choices=ATTRIBUTION_METHODS, help="Attribution method")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="With --all, run seeds 0..N-1 (overrides SEEDS from config)",
    )
    parser.add_argument("--all", action="store_true", help="Run all combinations")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing results/run_* dir; skip any (model,drift,method,seed) "
             "that already has a JSON checkpoint there and append to it",
    )
    args = parser.parse_args()

    if args.all:
        models = [args.model] if args.model else MODEL_TYPES
        drifts = [args.drift] if args.drift else DRIFT_TYPES
        attrs = [args.attribution] if args.attribution else ATTRIBUTION_METHODS
        if args.seed is not None:
            seeds = [args.seed]
        elif args.seeds is not None:
            seeds = list(range(args.seeds))
        else:
            seeds = SEEDS
        configs = [(m, d, a, s) for m in models for d in drifts for a in attrs for s in seeds]
    else:
        if not all([args.model, args.drift, args.attribution, args.seed is not None]):
            parser.error("Specify --model, --drift, --attribution, --seed or use --all")
        configs = [(args.model, args.drift, args.attribution, args.seed)]

    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            parser.error(f"--resume dir does not exist: {output_dir}")
        print(f"Resuming into existing dir: {output_dir}")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = RESULTS_DIR / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for model_type, drift_type, attr_method, seed in configs:
        seed_file = _seed_file(output_dir, model_type, drift_type, attr_method, seed)
        if seed_file.exists():
            print(f"\n[skip] {seed_file.name} already exists")
            results.append(json.loads(seed_file.read_text()))
            continue
        result = run_single_experiment(model_type, drift_type, attr_method, seed)
        seed_file.write_text(json.dumps(result, default=_json_default))
        print(f"  [checkpoint] {seed_file.name}")
        results.append(result)

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
