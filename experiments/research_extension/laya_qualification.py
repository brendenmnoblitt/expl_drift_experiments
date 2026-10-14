"""Bounded, endpoint-only Laya qualification on designated development data.

The runner never loads a local model. Remote calls require an explicit opt-in;
there are no retries, and each validated response is durably recorded at once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from experiments.laya_endpoint.client import EndpointClient, atomic_write_json, sha256_json

MODEL = {
    "repository": "convaiinnovations/laya",
    "revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
}
TASKS = {
    "youtube": {
        "instructions": "Which candidate best describes this YouTube comment?",
        "candidates": [
            {"candidate_id": "ham", "description": "A genuine, non-spam YouTube comment."},
            {
                "candidate_id": "spam",
                "description": "An unsolicited promotional or spam YouTube comment.",
            },
        ],
    },
    "ag_news": {
        "instructions": "Which candidate best describes the topic of this news article?",
        "candidates": [
            {
                "candidate_id": "world",
                "description": (
                    "World news: international affairs, politics, diplomacy, or conflict."
                ),
            },
            {
                "candidate_id": "sports",
                "description": "Sports news: athletes, teams, sporting events, or competitions.",
            },
            {
                "candidate_id": "business",
                "description": "Business news: companies, markets, finance, or the economy.",
            },
            {
                "candidate_id": "sci_tech",
                "description": (
                    "Science and technology news: research, scientific discoveries, computing, "
                    "or technological developments."
                ),
            },
        ],
    },
}
BATCH_SIZES = (1, 4, 8)
REPEATS = 2
MAX_EXAMPLES = 8
MAX_REQUESTS_PER_TARGET = sum(math.ceil(MAX_EXAMPLES / size) for size in BATCH_SIZES) * REPEATS
MAX_TIMEOUT_SECONDS = 900.0
MAX_WALL_SECONDS = 7200.0
MAX_GRADIENT_EVALUATIONS_PER_TARGET = 16384
MAX_TOTAL_GRADIENT_EVALUATIONS = (
    MAX_EXAMPLES * len(BATCH_SIZES) * REPEATS * 3 * MAX_GRADIENT_EVALUATIONS_PER_TARGET
)


def _write_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _samples_for_bank(
    development: dict[str, Any],
    response_parts: list[dict[str, Any]],
    candidate_ids: list[str],
    target_ids: list[str],
    reference_id: str,
) -> list[dict[str, Any]]:
    returned: dict[str, dict[str, Any]] = {}
    for response in response_parts:
        target_id = response["decision_target"]["target_candidate_id"]
        for sample in response["samples"]:
            row = returned.setdefault(sample["sample_id"], {"attributions": {}})
            if "scores" not in row:
                row.update(
                    {
                        "scores": sample["scores"],
                        "predicted_candidate_id": sample["predicted_candidate_id"],
                        "representation": sample["representation"],
                        "tokens": sample["tokens"],
                    }
                )
            elif any(
                row[key] != sample[source]
                for key, source in (
                    ("scores", "scores"),
                    ("predicted_candidate_id", "predicted_candidate_id"),
                    ("representation", "representation"),
                    ("tokens", "tokens"),
                )
            ):
                raise ValueError("scoring or token signals changed between fixed-target requests")
            row["attributions"][target_id] = sample["attribution"]

    normalized = []
    for source in development["samples"]:
        sample = returned[source["sample_id"]]
        if set(sample["attributions"]) != set(target_ids):
            raise ValueError("fixed-target requests did not cover every development sample")
        tokens = sample["tokens"]
        normalized.append(
            {
                "sample_id": source["sample_id"],
                "text_sha256": hashlib.sha256(source["text"].encode("utf-8")).hexdigest(),
                "candidate_ids": candidate_ids,
                "predicted_candidate_id": sample["predicted_candidate_id"],
                "logits": [score["logit"] for score in sample["scores"]],
                "probabilities": [score["probability"] for score in sample["scores"]],
                "representation": sample["representation"]["values"],
                "token_ids": tokens["token_ids"],
                "document_mask": [segment == "document" for segment in tokens["segments"]],
                "total_document_tokens": tokens["state_tokens_total"],
                "dropped_document_tokens": tokens["state_tokens_dropped"],
                "attributions": [
                    {
                        "target_candidate_id": target_id,
                        "reference_candidate_id": reference_id,
                        "values": sample["attributions"][target_id]["values"],
                        "input_target_value": sample["attributions"][target_id][
                            "input_target_value"
                        ],
                        "baseline_target_value": sample["attributions"][target_id][
                            "baseline_target_value"
                        ],
                        "completeness_delta": sample["attributions"][target_id][
                            "completeness_delta"
                        ],
                        "gradient_evaluations": sample["attributions"][target_id][
                            "gradient_evaluations"
                        ],
                    }
                    for target_id in target_ids
                ],
            }
        )
    return normalized


def _compare_banks(banks: list[dict[str, Any]], *, atol: float, rtol: float) -> dict[str, Any]:
    numeric_fields = ("logits", "probabilities", "representation")
    discrepancies = {key: 0.0 for key in (*numeric_fields, "attributions")}
    prediction_parity = True
    passed = True
    reference = banks[0]["samples"]
    for bank in banks:
        current = bank["samples"]
        if len(current) != len(reference):
            raise ValueError("qualification banks have different sample counts")
        for original, sample in zip(reference, current, strict=True):
            for identity in (
                "sample_id",
                "text_sha256",
                "candidate_ids",
                "token_ids",
                "document_mask",
                "total_document_tokens",
                "dropped_document_tokens",
            ):
                if original[identity] != sample[identity]:
                    raise ValueError(f"qualification changed input/token semantics: {identity}")
            prediction_parity &= (
                original["predicted_candidate_id"] == sample["predicted_candidate_id"]
            )
            for key in numeric_fields:
                a = np.asarray(original[key], dtype=np.float64)
                b = np.asarray(sample[key], dtype=np.float64)
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f"invalid qualification signal: {key}")
                discrepancies[key] = max(discrepancies[key], float(np.max(np.abs(a - b))))
                passed &= bool(np.allclose(a, b, atol=atol, rtol=rtol))
            if len(original["attributions"]) != len(sample["attributions"]):
                raise ValueError("qualification changed fixed-target count")
            for expected, actual in zip(
                original["attributions"], sample["attributions"], strict=True
            ):
                for identity in ("target_candidate_id", "reference_candidate_id"):
                    if expected[identity] != actual[identity]:
                        raise ValueError(f"qualification changed attribution target: {identity}")
                for key in (
                    "values",
                    "input_target_value",
                    "baseline_target_value",
                    "completeness_delta",
                ):
                    a = np.asarray(expected[key], dtype=np.float64)
                    b = np.asarray(actual[key], dtype=np.float64)
                    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise ValueError(f"invalid qualification attribution: {key}")
                    discrepancies["attributions"] = max(
                        discrepancies["attributions"], float(np.max(np.abs(a - b)))
                    )
                    passed &= bool(np.allclose(a, b, atol=atol, rtol=rtol))
    passed &= prediction_parity
    return {
        "atol": atol,
        "rtol": rtol,
        "numerical_comparison_passed": bool(passed),
        "prediction_parity_passed": bool(prediction_parity),
        "maximum_absolute_discrepancy": discrepancies,
        "scope": "these eight designated development examples and this exact endpoint runtime only",
    }


def qualify_laya(
    development: dict[str, Any],
    task: dict[str, Any],
    task_id: str,
    output: Path,
    *,
    endpoint_url: str,
    token: str,
    image_digest: str,
    allow_remote_execution: bool = False,
    timeout_seconds: float = MAX_TIMEOUT_SECONDS,
    wall_timeout_seconds: float = MAX_WALL_SECONDS,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    client_factory: Callable[..., EndpointClient] = EndpointClient,
) -> dict[str, Any]:
    """Compare repeated endpoint batches at sizes 1, 4 and 8; never retry a request."""
    if task_id not in TASKS or task != TASKS[task_id]:
        raise ValueError("task ID and task definition must match a pinned Laya task")
    if not allow_remote_execution:
        raise ValueError(
            "remote endpoint execution requires explicit approval via --allow-remote-execution"
        )
    if output.exists():
        raise ValueError("choose a fresh remote qualification directory; resume is not permitted")
    if development.get("role") != "development":
        raise ValueError("qualification requires designated development; held-out data is sealed")
    samples = development.get("samples")
    if not isinstance(samples, list) or len(samples) != MAX_EXAMPLES:
        raise ValueError(
            f"remote qualification is fixed at exactly {MAX_EXAMPLES} development examples"
        )
    sample_ids = [sample.get("sample_id") for sample in samples]
    if len(set(sample_ids)) != MAX_EXAMPLES or any(not sample.get("text") for sample in samples):
        raise ValueError("development sample IDs must be unique and texts non-empty")
    if (
        not np.isfinite([atol, rtol, timeout_seconds, wall_timeout_seconds]).all()
        or min(atol, rtol) < 0
        or not (0 < timeout_seconds <= MAX_TIMEOUT_SECONDS)
        or not (0 < wall_timeout_seconds <= MAX_WALL_SECONDS)
    ):
        raise ValueError("tolerances or bounded request/wall timeouts are invalid")
    if not endpoint_url or not token or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
        raise ValueError("endpoint URL, token and immutable SHA-256 image digest are required")
    task_candidates = task.get("candidates")
    if not isinstance(task_candidates, list) or any(
        not isinstance(candidate, dict) for candidate in task_candidates
    ):
        raise ValueError("task candidate list is required and every candidate must be an object")
    candidate_ids = [candidate.get("candidate_id") for candidate in task_candidates]
    reference_id = development.get("task", {}).get("reference_candidate_id")
    if (
        not task.get("instructions")
        or len(candidate_ids) not in (2, 4)
        or len(set(candidate_ids)) != len(candidate_ids)
        or reference_id not in candidate_ids
        or candidate_ids != development.get("task", {}).get("candidate_ids")
    ):
        raise ValueError("task must match the designated ordered two- or four-candidate schema")
    target_ids = [candidate for candidate in candidate_ids if candidate != reference_id]
    expected_requests = MAX_REQUESTS_PER_TARGET * len(target_ids)
    total_gradient_budget = (
        MAX_EXAMPLES
        * len(BATCH_SIZES)
        * REPEATS
        * len(target_ids)
        * MAX_GRADIENT_EVALUATIONS_PER_TARGET
    )

    output.mkdir(parents=True)
    started = time.monotonic()
    client = client_factory(endpoint_url, token, timeout=min(timeout_seconds, MAX_TIMEOUT_SECONDS))
    try:
        health = client.health()
        if health.get("schema_version") != "4" or health.get("model") != MODEL:
            raise RuntimeError(
                "endpoint health reports a different schema or pinned model revision"
            )
        runtime = health.get("runtime")
        if not isinstance(runtime, dict) or runtime.get("configured_image_digest") != image_digest:
            raise RuntimeError(
                "endpoint runtime image identity differs from the requested immutable digest"
            )
        required = {
            "integrated_gradients",
            "fixed_candidate_margin",
            "document_mean_representation",
        }
        if not required.issubset(set(health.get("capabilities", []))):
            raise RuntimeError("endpoint health omits required implemented signals")

        plan = {
            "schema_version": 2,
            "role": "development",
            "task_id": task_id,
            "development_sha256": sha256_json(development),
            "task_sha256": sha256_json(task),
            "model": MODEL,
            "runtime": runtime,
            "configured_image_digest": image_digest,
            "candidate_ids": candidate_ids,
            "reference_candidate_id": reference_id,
            "target_candidate_ids": target_ids,
            "batch_sizes": list(BATCH_SIZES),
            "repeats": REPEATS,
            "max_examples": MAX_EXAMPLES,
            "max_requests": expected_requests,
            "max_requests_per_target": MAX_REQUESTS_PER_TARGET,
            "max_request_timeout_seconds": timeout_seconds,
            "max_total_wall_seconds": wall_timeout_seconds,
            "max_gradient_evaluations_per_target": MAX_GRADIENT_EVALUATIONS_PER_TARGET,
            "max_total_gradient_evaluations": total_gradient_budget,
            "max_tokens": 128,
            "n_steps_requested": 64,
            "atol": atol,
            "rtol": rtol,
            "retry_policy": "none; failed requests are recorded and not resumed",
        }
        atomic_write_json(output / "comparison-plan.json", plan)
        banks = []
        request_index = 0
        for batch_size in BATCH_SIZES:
            for repeat in range(REPEATS):
                responses = []
                for target_id in target_ids:
                    decision_target = {
                        "target_semantics": "fixed_candidate_margin",
                        "target_candidate_id": target_id,
                        "reference_candidate_id": reference_id,
                    }
                    for chunk_index, offset in enumerate(range(0, MAX_EXAMPLES, batch_size)):
                        chunk = samples[offset : offset + batch_size]
                        endpoint_samples = [
                            {"sample_id": sample["sample_id"], "text": sample["text"]}
                            for sample in chunk
                        ]
                        corpus_sha = sha256_json(endpoint_samples)
                        request = {
                            "schema_version": "4",
                            "request_id": (
                                f"laya-qualification-{task_id}-{target_id}-{batch_size}-r{repeat}-"
                                f"c{chunk_index}-{corpus_sha[:12]}"
                            ),
                            "run_id": f"laya-development-qualification-{task_id}-v1",
                            "window_index": 0,
                            "corpus_sha256": corpus_sha,
                            "model": MODEL,
                            "tokenizer": MODEL,
                            "instructions": task["instructions"],
                            "candidates": task_candidates,
                            "samples": endpoint_samples,
                            "max_tokens": 128,
                            "decision_target": decision_target,
                            "precision": "float32_no_autocast",
                            "tokenization": "laya_sdk_0.3.29_encode_state",
                            "pooling": "document_mean_last_hidden_state",
                            "runtime": runtime,
                            "attribution": {"method": "integrated_gradients", "n_steps": 64},
                            "representations": True,
                        }
                        request_index += 1
                        if request_index > expected_requests:
                            raise RuntimeError("qualification request ceiling exceeded")
                        request_sha = sha256_json(request)
                        request_path = output / "requests" / f"request-{request_index:02d}.json"
                        atomic_write_json(
                            request_path, {"request_sha256": request_sha, "request": request}
                        )
                        remaining = wall_timeout_seconds - (time.monotonic() - started)
                        if remaining <= 0:
                            raise TimeoutError("qualification total wall-time ceiling reached")
                        client.timeout = min(timeout_seconds, remaining)
                        request_started = time.monotonic()
                        try:
                            response = client.extract(request)
                        except Exception as error:
                            _write_event(
                                output / "events.jsonl",
                                {
                                    "request_index": request_index,
                                    "request_id": request["request_id"],
                                    "request_sha256": request_sha,
                                    "status": "failed",
                                    "elapsed_seconds": time.monotonic() - request_started,
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                },
                            )
                            raise
                        elapsed = time.monotonic() - request_started
                        record = {
                            "request_sha256": request_sha,
                            "endpoint_image_digest": image_digest,
                            "response": response,
                        }
                        atomic_write_json(
                            output / "responses" / f"response-{request_index:02d}.json", record
                        )
                        responses.append(response)
                        _write_event(
                            output / "events.jsonl",
                            {
                                "request_index": request_index,
                                "request_id": request["request_id"],
                                "request_sha256": request_sha,
                                "response_sha256": sha256_json(response),
                                "status": "completed",
                                "elapsed_seconds": elapsed,
                                "endpoint_latency_seconds": response["latency_seconds"],
                                "target_candidate_id": target_id,
                                "gradient_evaluations": sum(
                                    row["attribution"]["gradient_evaluations"]
                                    for row in response["samples"]
                                ),
                            },
                        )
                        print(
                            json.dumps(
                                {
                                    "request": request_index,
                                    "maximum": expected_requests,
                                    "target": target_id,
                                    "status": "completed",
                                }
                            ),
                            flush=True,
                        )
                bank = {
                    "task_id": task_id,
                    "batch_size": batch_size,
                    "repeat": repeat,
                    "runtime": runtime,
                    "response_parts": responses,
                    "samples": _samples_for_bank(
                        development, responses, candidate_ids, target_ids, reference_id
                    ),
                }
                atomic_write_json(output / f"batch-{batch_size}-repeat-{repeat}.json", bank)
                banks.append(bank)

        comparison = _compare_banks(banks, atol=atol, rtol=rtol)
        gradient_evaluations = [
            attribution["gradient_evaluations"]
            for bank in banks
            for sample in bank["samples"]
            for attribution in sample["attributions"]
        ]
        report = {
            "schema_version": 2,
            "role": "development",
            "task_id": task_id,
            "plan_sha256": sha256_json(plan),
            "runtime": runtime,
            "comparison": comparison,
            "bank_sha256": [sha256_json(bank) for bank in banks],
            "measured_cost": {
                "requests_completed": request_index,
                "source_examples": MAX_EXAMPLES,
                "extracted_examples": sum(len(bank["samples"]) for bank in banks),
                "fixed_targets_per_example": len(target_ids),
                "actual_gradient_evaluations": sum(gradient_evaluations),
                "max_evaluations_per_target": max(gradient_evaluations),
                "max_total_gradient_evaluations": total_gradient_budget,
                "endpoint_inference_seconds": sum(
                    event["endpoint_latency_seconds"]
                    for event in _read_events(output / "events.jsonl")
                ),
                "wall_seconds": time.monotonic() - started,
            },
            "G1": (
                "not_established: bounded development comparison only; full case coverage "
                "and resource approval remain required"
            ),
        }
        atomic_write_json(output / "qualification.json", report)
        return report
    except Exception:
        if not (output / "qualification.json").exists():
            atomic_write_json(
                output / "run-failure.json",
                {"status": "incomplete", "elapsed_seconds": time.monotonic() - started},
            )
        raise


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--development-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-remote-execution", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=MAX_TIMEOUT_SECONDS)
    parser.add_argument("--wall-timeout-seconds", type=float, default=MAX_WALL_SECONDS)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    report = qualify_laya(
        json.loads(args.development_json.read_text(encoding="utf-8")),
        TASKS[args.task],
        args.task,
        args.output,
        endpoint_url=os.environ.get("LAYA_ENDPOINT_URL", ""),
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN", ""),
        image_digest=os.environ.get("LAYA_ENDPOINT_IMAGE_DIGEST", ""),
        allow_remote_execution=args.allow_remote_execution,
        timeout_seconds=args.timeout_seconds,
        wall_timeout_seconds=args.wall_timeout_seconds,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(json.dumps(report, indent=2))
    if not report["comparison"]["numerical_comparison_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
