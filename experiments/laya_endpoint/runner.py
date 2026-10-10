"""Run a small, host-orchestrated drift experiment against the remote Laya endpoint."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np

from .client import EndpointClient, atomic_write_json, load_cached_response, sha256_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DATASET_URL = "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip"
DATASET_SHA256 = "bd6182891adb3cfc8334b82c062176dfbebc563bf0ba07e31c2645f916865a0a"
DATASET_LICENSE = "CC BY 4.0"
MODEL = {
    "repository": "convaiinnovations/laya",
    "revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
}
CANDIDATES = [
    {"candidate_id": "ham", "description": "A genuine, non-spam YouTube comment."},
    {"candidate_id": "spam", "description": "An unsolicited promotional or spam YouTube comment."},
]
INSTRUCTIONS = "Which candidate best describes this YouTube comment?"
SEED = 42
MAX_TOKENS = 128
WINDOW_CLASS_COUNTS = [
    {0: 2, 1: 2},
    {0: 2, 1: 2},
    {0: 2, 1: 2},
    {0: 1, 1: 3},
    {0: 0, 1: 4},
    {0: 0, 1: 4},
]
CALIBRATION_WINDOWS = 2


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_dataset() -> tuple[bytes, list[dict[str, Any]]]:
    with urlopen(DATASET_URL, timeout=60) as response:
        archive = response.read()
    digest = hashlib.sha256(archive).hexdigest()
    if digest != DATASET_SHA256:
        raise RuntimeError(f"UCI dataset SHA-256 changed: expected {DATASET_SHA256}, got {digest}")

    records = []
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for filename in sorted(bundle.namelist()):
            if not filename.endswith(".csv") or filename.startswith("__MACOSX/"):
                continue
            video = Path(filename).stem
            with bundle.open(filename) as raw:
                rows = csv.DictReader(io.TextIOWrapper(raw, encoding="latin-1"))
                for row in rows:
                    text = row["CONTENT"].strip()
                    label = int(row["CLASS"])
                    if text and label in (0, 1):
                        records.append(
                            {
                                "sample_id": f"{video}-{row['COMMENT_ID']}",
                                "text": text,
                                "label": label,
                            }
                        )
    if len(records) != 1956:
        raise RuntimeError(f"Expected 1956 UCI comments, found {len(records)}")
    return archive, records


def _create_windows(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rng = random.Random(SEED)
    by_label = {label: [row for row in records if row["label"] == label] for label in (0, 1)}
    for pool in by_label.values():
        rng.shuffle(pool)
    offsets = {0: 0, 1: 0}
    windows = []
    for counts in WINDOW_CLASS_COUNTS:
        window = []
        for label in (0, 1):
            start = offsets[label]
            stop = start + counts[label]
            window.extend(by_label[label][start:stop])
            offsets[label] = stop
        rng.shuffle(window)
        windows.append(window)
    return windows


def _request_for_window(
    run_id: str, window_index: int, samples: list[dict[str, Any]]
) -> dict[str, Any]:
    endpoint_samples = [{"sample_id": row["sample_id"], "text": row["text"]} for row in samples]
    corpus_fingerprint = sha256_json(endpoint_samples)
    return {
        "schema_version": "1",
        "request_id": f"{run_id}-w{window_index}-{corpus_fingerprint[:12]}",
        "run_id": run_id,
        "window_index": window_index,
        "corpus_sha256": corpus_fingerprint,
        "model": MODEL,
        "tokenizer": MODEL,
        "instructions": INSTRUCTIONS,
        "candidates": CANDIDATES,
        "samples": endpoint_samples,
        "max_tokens": MAX_TOKENS,
        "attribution": {
            "method": "integrated_gradients",
            "target_candidate_id": "spam",
            "n_steps": 64,
        },
        "representations": False,
    }


def _atomic_write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fields = list(dict.fromkeys(key for record in records for key in record))
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(output_dir: Path) -> dict[str, Any]:
    endpoint_url = os.environ["LAYA_ENDPOINT_URL"]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN for the private HF endpoint")
    image_digest = os.environ["LAYA_ENDPOINT_IMAGE_DIGEST"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
        raise ValueError("LAYA_ENDPOINT_IMAGE_DIGEST must be an immutable sha256 digest")

    root = REPOSITORY_ROOT
    core_repository = root.parent / "expl_drift"
    sys.path.insert(0, str(root.parent))
    from expl_drift import DriftDetector, DriftMonitor
    from expl_drift.drift.summarize import summarize_attributions

    run_id = "laya-youtube-spam-seed-42"
    archive, records = _load_dataset()
    windows = _create_windows(records)
    dataset_fingerprint = hashlib.sha256(archive).hexdigest()
    provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": {
            "name": "UCI YouTube Spam Collection",
            "url": DATASET_URL,
            "license": DATASET_LICENSE,
            "citation": "Alberto & Lochter (2015), UCI DOI 10.24432/C58885",
            "archive_sha256": dataset_fingerprint,
            "records": len(records),
            "training_mix_overlap": "not verifiable; Laya's full training mix is undisclosed",
            "published_laya_benchmarks_overlap": "not listed in the pinned model card",
        },
        "model": MODEL,
        "tokenizer": MODEL,
        "endpoint": {"url": endpoint_url, "image_digest": image_digest},
        "revisions": {
            "expl_drift_experiments": _git_revision(root),
            "expl_drift": _git_revision(core_repository),
        },
        "experiment": {
            "seed": SEED,
            "windows": len(windows),
            "samples_per_window": 4,
            "calibration_windows": CALIBRATION_WINDOWS,
            "window_class_counts": [
                {"ham": counts[0], "spam": counts[1]} for counts in WINDOW_CLASS_COUNTS
            ],
            "max_tokens": MAX_TOKENS,
            "attribution": "integrated_gradients target=spam candidate_logit n_steps=64",
            "drift_injection": (
                "synthetic class-prior shift by deterministic sampling without replacement"
            ),
            "drift_method": "DriftDetector + DriftMonitor + summarize_attributions",
            "bert_gpt_comparison": (
                "Existing BERT/GPT runs use AG News and different labels; results are context, "
                "not a direct baseline for this corpus."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "provenance.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != provenance:
            raise ValueError("output directory provenance differs; choose a fresh output directory")
    else:
        atomic_write_json(manifest_path, provenance)

    client = EndpointClient(endpoint_url, token)
    health = client.health()
    if health.get("model") != MODEL:
        raise RuntimeError("endpoint health reports a different model revision")
    if "integrated_gradients" not in health.get("capabilities", []):
        raise RuntimeError("endpoint health does not advertise qualified integrated_gradients")

    responses = []
    cache_dir = output_dir / "windows"
    for window_index, samples in enumerate(windows):
        request = _request_for_window(run_id, window_index, samples)
        request_digest = sha256_json(request)
        cache_path = cache_dir / f"window-{window_index:02d}.json"
        expected_labels = {row["sample_id"]: row["label"] for row in samples}
        if cache_path.exists():
            cached_entry = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached_entry.get("labels") != expected_labels:
                raise ValueError(f"cached labels differ for window {window_index}")
        response = load_cached_response(cache_path, request)
        if response is None:
            response = client.extract(request)
            atomic_write_json(
                cache_path,
                {
                    "request_sha256": request_digest,
                    "labels": expected_labels,
                    "response": response,
                },
            )
        responses.append(response)

    raw_attributions = [
        np.asarray([sample["attribution"]["values"] for sample in response["samples"]], dtype=float)
        for response in responses
    ]
    summaries = [summarize_attributions(values) for values in raw_attributions]
    detector = DriftDetector(summaries[0])
    monitor = DriftMonitor(
        detector,
        summaries[1 : 1 + CALIBRATION_WINDOWS],
        warning_std=2.5,
        critical_std=3.5,
    )

    rows = []
    for window_index, (source, response) in enumerate(zip(windows, responses, strict=True)):
        labels = {row["sample_id"]: row["label"] for row in source}
        correct = sum(
            int(
                sample["predicted_candidate_id"]
                == ("spam" if labels[sample["sample_id"]] else "ham")
            )
            for sample in response["samples"]
        )
        row = {
            "window": window_index,
            "ham": sum(row["label"] == 0 for row in source),
            "spam": sum(row["label"] == 1 for row in source),
            "accuracy": correct / len(source),
            "alert_level": "calibration",
        }
        if window_index > CALIBRATION_WINDOWS:
            evaluation = monitor.evaluate(summaries[window_index])
            row["alert_level"] = evaluation["alert_level"].value
            row.update(evaluation["metrics"])
        rows.append(row)

    result = {
        "provenance": provenance,
        "endpoint_health": health,
        "window_results": rows,
        "summary": {
            "baseline_accuracy": rows[0]["accuracy"],
            "mean_later_accuracy": float(np.mean([row["accuracy"] for row in rows[1:]])),
            "first_drift_window": CALIBRATION_WINDOWS + 1,
            "alert_levels": [row["alert_level"] for row in rows],
        },
    }
    atomic_write_json(output_dir / "experiment_results.json", result)
    _atomic_write_csv(output_dir / "window_results.csv", rows)
    return result


def main() -> None:
    output = Path(os.environ.get("LAYA_EXPERIMENT_OUTPUT", "results/laya-youtube-spam"))
    result = run(output)
    print(json.dumps(result["summary"], sort_keys=True))
    for row in result["window_results"]:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
