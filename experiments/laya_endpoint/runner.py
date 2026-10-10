"""Run a small, host-orchestrated drift experiment against the remote Laya endpoint."""

from __future__ import annotations

import argparse
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
MAX_BATCH_SAMPLES = 32
WARNING_STD = 2.5
CRITICAL_STD = 3.5
WINDOW_CLASS_COUNTS = [
    {0: 2, 1: 2},
    {0: 2, 1: 2},
    {0: 2, 1: 2},
    {0: 1, 1: 3},
    {0: 0, 1: 4},
    {0: 0, 1: 4},
]
CALIBRATION_WINDOWS = 2
CALIBRATED_WINDOW_CLASS_COUNTS = [
    {0: 50, 1: 50},
    {0: 50, 1: 50},
    {0: 50, 1: 50},
    {0: 50, 1: 50},
    {0: 50, 1: 50},
    {0: 40, 1: 60},
    {0: 30, 1: 70},
    {0: 20, 1: 80},
    {0: 10, 1: 90},
    {0: 0, 1: 100},
]
EXPERIMENT_PROFILES = {
    "pilot": {
        "run_id": "laya-youtube-spam-seed-42",
        "window_class_counts": WINDOW_CLASS_COUNTS,
        "calibration_windows": CALIBRATION_WINDOWS,
    },
    "calibrated": {
        "run_id": "laya-youtube-spam-calibrated-seed-42",
        "window_class_counts": CALIBRATED_WINDOW_CLASS_COUNTS,
        "calibration_windows": 4,
    },
}


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _merge_resume_provenance(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    resume_from_image_digest: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Track every runtime/code revision used across an explicitly resumed run."""
    previous_endpoint = previous["endpoint"]
    current_endpoint = current["endpoint"]
    previous_image = previous_endpoint["image_digest"]
    current_image = current_endpoint["image_digest"]
    image_digests = list(previous_endpoint.get("image_digests", [previous_image]))
    if current_image not in image_digests:
        if resume_from_image_digest != previous_image:
            raise ValueError(
                "endpoint image changed; set LAYA_RESUME_PREVIOUS_IMAGE_DIGEST "
                "to the image recorded in the existing provenance to resume"
            )
        image_digests.append(current_image)

    def without_execution_revisions(value: dict[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(value))
        for field in ("image_digest", "image_digests", "legacy_cache_image_digest"):
            normalized["endpoint"].pop(field, None)
        for field in ("expl_drift_experiments", "experiment_code_revisions"):
            normalized["revisions"].pop(field, None)
        return normalized

    if without_execution_revisions(previous) != without_execution_revisions(current):
        raise ValueError("output directory provenance differs; choose a fresh output directory")

    previous_revisions = previous["revisions"]
    current_revisions = current["revisions"]
    code_revisions = previous_revisions.get(
        "experiment_code_revisions", [previous_revisions["expl_drift_experiments"]]
    )
    code_revisions = list(
        dict.fromkeys([*code_revisions, *current_revisions["experiment_code_revisions"]])
    )
    current_endpoint["image_digests"] = image_digests
    legacy_cache_image_digest = previous_endpoint.get("legacy_cache_image_digest", previous_image)
    if len(image_digests) > 1 or "legacy_cache_image_digest" in previous_endpoint:
        current_endpoint["legacy_cache_image_digest"] = legacy_cache_image_digest
    current_revisions["expl_drift_experiments"] = code_revisions[0]
    current_revisions["experiment_code_revisions"] = code_revisions
    return current, legacy_cache_image_digest


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


def _create_windows(
    records: list[dict[str, Any]],
    class_counts: list[dict[int, int]] | None = None,
) -> list[list[dict[str, Any]]]:
    rng = random.Random(SEED)
    by_label = {label: [row for row in records if row["label"] == label] for label in (0, 1)}
    for pool in by_label.values():
        rng.shuffle(pool)
    offsets = {0: 0, 1: 0}
    windows = []
    for window_index, counts in enumerate(class_counts or WINDOW_CLASS_COUNTS):
        window = []
        for label in (0, 1):
            start = offsets[label]
            stop = start + counts[label]
            if stop > len(by_label[label]):
                raise ValueError(
                    f"window {window_index} requests {stop} samples of label {label}; "
                    f"dataset contains {len(by_label[label])}"
                )
            window.extend(by_label[label][start:stop])
            offsets[label] = stop
        rng.shuffle(window)
        windows.append(window)
    return windows


def _window_chunks(samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        samples[start : start + MAX_BATCH_SAMPLES]
        for start in range(0, len(samples), MAX_BATCH_SAMPLES)
    ]


def _request_for_window(
    run_id: str,
    window_index: int,
    samples: list[dict[str, Any]],
    *,
    chunk_index: int = 0,
    chunk_count: int = 1,
) -> dict[str, Any]:
    endpoint_samples = [{"sample_id": row["sample_id"], "text": row["text"]} for row in samples]
    corpus_fingerprint = sha256_json(endpoint_samples)
    chunk_tag = f"-b{chunk_index}" if chunk_count > 1 else ""
    return {
        "schema_version": "1",
        "request_id": f"{run_id}-w{window_index}{chunk_tag}-{corpus_fingerprint[:12]}",
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


def run(output_dir: Path, profile: str = "pilot") -> dict[str, Any]:
    try:
        profile_config = EXPERIMENT_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"unknown experiment profile {profile!r}") from error

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

    run_id = profile_config["run_id"]
    window_class_counts = profile_config["window_class_counts"]
    calibration_windows = profile_config["calibration_windows"]
    archive, records = _load_dataset()
    windows = _create_windows(records, window_class_counts)
    dataset_fingerprint = hashlib.sha256(archive).hexdigest()
    experiment_revision = _git_revision(root)
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
        "endpoint": {
            "url": endpoint_url,
            "image_digest": image_digest,
            "image_digests": [image_digest],
        },
        "revisions": {
            "expl_drift_experiments": experiment_revision,
            "experiment_code_revisions": [experiment_revision],
            "expl_drift": _git_revision(core_repository),
        },
        "experiment": {
            "profile": profile,
            "seed": SEED,
            "windows": len(windows),
            "samples_per_window": len(windows[0]),
            "calibration_windows": calibration_windows,
            "warning_std": WARNING_STD,
            "critical_std": CRITICAL_STD,
            "window_class_counts": [
                {"ham": counts[0], "spam": counts[1]} for counts in window_class_counts
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
    legacy_cache_image_digest = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance, legacy_cache_image_digest = _merge_resume_provenance(
            previous,
            provenance,
            resume_from_image_digest=os.environ.get("LAYA_RESUME_PREVIOUS_IMAGE_DIGEST"),
        )
        if previous != provenance:
            atomic_write_json(manifest_path, provenance)
    else:
        atomic_write_json(manifest_path, provenance)

    client = EndpointClient(endpoint_url, token)
    health = client.health()
    if health.get("model") != MODEL:
        raise RuntimeError("endpoint health reports a different model revision")
    if "integrated_gradients" not in health.get("capabilities", []):
        raise RuntimeError("endpoint health does not advertise integrated_gradients")

    responses = []
    cache_dir = output_dir / "windows"
    cache_image_digests = set()
    for window_index, samples in enumerate(windows):
        chunks = _window_chunks(samples)
        window_samples = []
        for chunk_index, chunk in enumerate(chunks):
            request = _request_for_window(
                run_id,
                window_index,
                chunk,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
            )
            cache_path = (
                cache_dir / f"window-{window_index:02d}.json"
                if len(chunks) == 1
                else cache_dir / f"window-{window_index:02d}-chunk-{chunk_index:02d}.json"
            )
            expected_labels = {row["sample_id"]: row["label"] for row in chunk}
            cache_image_digest = image_digest
            if cache_path.exists():
                cached_entry = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached_entry.get("labels") != expected_labels:
                    raise ValueError(f"cached labels differ for window {window_index}")
                cache_image_digest = cached_entry.get("endpoint_image_digest")
                if cache_image_digest is None:
                    cache_image_digest = legacy_cache_image_digest
                    if cache_image_digest is None:
                        raise ValueError(f"cache {cache_path} has no endpoint image provenance")
                    cached_entry["endpoint_image_digest"] = cache_image_digest
                    atomic_write_json(cache_path, cached_entry)
                if cache_image_digest not in provenance["endpoint"]["image_digests"]:
                    raise ValueError(f"cache {cache_path} references an unrecorded endpoint image")
                cache_image_digests.add(cache_image_digest)
            response = load_cached_response(cache_path, request)
            if response is None:
                print(
                    f"Requesting window {window_index + 1}/{len(windows)} "
                    f"batch {chunk_index + 1}/{len(chunks)}: {request['request_id']}",
                    flush=True,
                )
                try:
                    response = client.extract(request)
                except Exception as error:
                    raise RuntimeError(
                        f"endpoint extraction failed for window {window_index} "
                        f"batch {chunk_index} ({request['request_id']})"
                    ) from error
                cache_image_digests.add(image_digest)
                atomic_write_json(
                    cache_path,
                    {
                        "request_sha256": sha256_json(request),
                        "labels": expected_labels,
                        "endpoint_image_digest": image_digest,
                        "response": response,
                    },
                )
                print(
                    f"Validated and cached window {window_index + 1}/{len(windows)} "
                    f"batch {chunk_index + 1}/{len(chunks)}",
                    flush=True,
                )
            else:
                print(
                    f"Reusing validated window {window_index + 1}/{len(windows)} "
                    f"batch {chunk_index + 1}/{len(chunks)}",
                    flush=True,
                )
            window_samples.extend(response["samples"])
        responses.append({"samples": window_samples})

    raw_attributions = [
        np.asarray([sample["attribution"]["values"] for sample in response["samples"]], dtype=float)
        for response in responses
    ]
    summaries = [summarize_attributions(values) for values in raw_attributions]
    detector = DriftDetector(summaries[0])
    monitor = DriftMonitor(
        detector,
        summaries[1 : 1 + calibration_windows],
        warning_std=WARNING_STD,
        critical_std=CRITICAL_STD,
    )

    rows = []
    for window_index, (source, response) in enumerate(zip(windows, responses, strict=True)):
        labels = {row["sample_id"]: row["label"] for row in source}
        spam_probabilities = [
            next(
                score["probability"]
                for score in sample["scores"]
                if score["candidate_id"] == "spam"
            )
            for sample in response["samples"]
        ]
        correct = sum(
            int(
                sample["predicted_candidate_id"]
                == ("spam" if labels[sample["sample_id"]] else "ham")
            )
            for sample in response["samples"]
        )
        row = {
            "window": window_index,
            "ham": sum(sample["label"] == 0 for sample in source),
            "spam": sum(sample["label"] == 1 for sample in source),
            "observed_spam_fraction": sum(sample["label"] == 1 for sample in source) / len(source),
            "predicted_spam_fraction": sum(
                sample["predicted_candidate_id"] == "spam" for sample in response["samples"]
            )
            / len(source),
            "mean_spam_probability": float(np.mean(spam_probabilities)),
            "accuracy": correct / len(source),
            "alert_level": "calibration",
        }
        if window_index > calibration_windows:
            evaluation = monitor.evaluate(summaries[window_index])
            row["alert_level"] = evaluation["alert_level"].value
            row.update(evaluation["metrics"])
            row.update(
                {f"{metric}_alert": alert.value for metric, alert in evaluation["alerts"].items()}
            )
        rows.append(row)

    result = {
        "provenance": provenance,
        "endpoint_health": health,
        "endpoint_image_digests_used": sorted(cache_image_digests),
        "monitor_thresholds": monitor.thresholds,
        "window_results": rows,
        "summary": {
            "baseline_accuracy": rows[0]["accuracy"],
            "mean_later_accuracy": float(np.mean([row["accuracy"] for row in rows[1:]])),
            "mean_post_calibration_accuracy": float(
                np.mean([row["accuracy"] for row in rows[calibration_windows + 1 :]])
            ),
            "first_drift_window": calibration_windows + 1,
            "alert_levels": [row["alert_level"] for row in rows],
        },
    }
    atomic_write_json(output_dir / "experiment_results.json", result)
    _atomic_write_csv(output_dir / "window_results.csv", rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(EXPERIMENT_PROFILES), default="pilot")
    args = parser.parse_args()
    default_output = f"results/laya-youtube-spam-{args.profile}"
    output = Path(os.environ.get("LAYA_EXPERIMENT_OUTPUT", default_output))
    result = run(output, profile=args.profile)
    print(json.dumps(result["summary"], sort_keys=True))
    for row in result["window_results"]:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
