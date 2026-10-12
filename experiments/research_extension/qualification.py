"""Bounded reference-model development extraction and batch-invariance reports.

Only designated development inputs are accepted. Passing these numerical checks
is not the full G1 gate and does not authorize reference/tuning or primary runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .data import save_artifact, sha256_json
from .reference_adapter import ReferenceAdapter


def compare_banks(banks: list[dict[str, Any]], *, atol: float, rtol: float) -> dict[str, Any]:
    """Compare actual scores, pooled signals, signed IG and singleton/batch parity."""
    if not banks or not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("banks and prespecified finite nonnegative tolerances are required")
    first = banks[0]
    discrepancies = {
        key: 0.0
        for key in (
            "logits",
            "probabilities",
            "representation",
            "attributions",
            "singleton_vs_batch_logits",
        )
    }
    passed = True
    for bank in banks:
        if bank["runtime"] != first["runtime"] or len(bank["samples"]) != len(first["samples"]):
            raise ValueError("qualification banks have different runtimes or sample counts")
        for original, sample in zip(first["samples"], bank["samples"], strict=True):
            for identity in (
                "sample_id",
                "text_sha256",
                "candidate_ids",
                "token_ids",
                "document_mask",
                "total_document_tokens",
                "dropped_document_tokens",
            ):
                if sample[identity] != original[identity]:
                    raise ValueError(f"qualification changed input/token semantics: {identity}")
            for key in ("logits", "probabilities", "representation"):
                a, b = np.asarray(original[key]), np.asarray(sample[key])
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f"invalid qualification numerical signal: {key}")
                discrepancies[key] = max(discrepancies[key], float(np.max(np.abs(a - b))))
                passed &= bool(np.allclose(a, b, atol=atol, rtol=rtol))
            singleton, batch = (
                np.asarray(sample["attribution_input_logits"]),
                np.asarray(sample["logits"]),
            )
            discrepancies["singleton_vs_batch_logits"] = max(
                discrepancies["singleton_vs_batch_logits"], float(np.max(np.abs(singleton - batch)))
            )
            passed &= bool(np.allclose(singleton, batch, atol=atol, rtol=rtol))
            if len(sample["attributions"]) != len(original["attributions"]):
                raise ValueError("qualification changed fixed target count")
            for a, b in zip(original["attributions"], sample["attributions"], strict=True):
                for identity in (
                    "target_candidate_id",
                    "reference_candidate_id",
                    "target_semantics",
                ):
                    if a[identity] != b[identity]:
                        raise ValueError("qualification changed fixed decision target")
                for signal in (a, b):
                    delta = signal["input_target_value"] - signal["baseline_target_value"]
                    residual = delta - math.fsum(signal["values"])
                    if not np.isclose(residual, signal["completeness_delta"], atol=1e-6, rtol=1e-6):
                        raise ValueError("saved reference completeness cannot be recalculated")
                    tolerance = bank["runtime"]["completeness_atol"] + bank["runtime"][
                        "completeness_rtol"
                    ] * abs(delta)
                    passed &= abs(residual) <= tolerance
                av, bv = np.asarray(a["values"]), np.asarray(b["values"])
                if av.shape != bv.shape or not np.isfinite(av).all() or not np.isfinite(bv).all():
                    raise ValueError("invalid signed qualification arrays")
                discrepancies["attributions"] = max(
                    discrepancies["attributions"], float(np.max(np.abs(av - bv)))
                )
                passed &= bool(np.allclose(av, bv, atol=atol, rtol=rtol))
    return {
        "atol": atol,
        "rtol": rtol,
        "numerical_comparison_passed": bool(passed),
        "maximum_absolute_discrepancy": discrepancies,
        "scope": (
            "these designated examples/runtimes only; not full task qualification "
            "or a permission to reuse arbitrary batch contexts"
        ),
    }


def qualify_reference(
    checkpoint: Path,
    development: dict[str, Any],
    output: Path,
    *,
    device: str,
    max_tokens: int,
    n_steps: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Save six bounded banks: repeated requests at batch sizes 1, 4 and 8."""
    if output.exists() and any(output.glob("batch-*.json")):
        raise ValueError("qualification output already contains extraction; choose a fresh run")
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("prespecified tolerances must be finite and nonnegative")
    if development.get("role") != "development":
        raise ValueError(
            "qualification requires a designated development allocation; held-out data is sealed"
        )
    settings = development["task"]
    samples = development["samples"]
    if len(samples) < 8:
        raise ValueError("at least eight development samples are needed for the batch-8 comparison")
    identity = sha256_json(development)
    # Persist tolerances before any forward pass. Incompatible reuse is refused.
    save_artifact(
        output / "comparison-plan.json",
        {
            "schema_version": 1,
            "development_sha256": identity,
            "batch_sizes": [1, 4, 8],
            "repeats": 2,
            "atol": atol,
            "rtol": rtol,
            "max_tokens": max_tokens,
            "n_steps": n_steps,
            "checkpoint": str(checkpoint.resolve()),
            "device": device,
        },
    )
    adapter = ReferenceAdapter.from_checkpoint(
        checkpoint,
        candidate_ids=settings["candidate_ids"],
        reference_candidate_id=settings["reference_candidate_id"],
        device=device,
        max_tokens=max_tokens,
        n_steps=n_steps,
    )
    banks = []
    for size in (1, 4, 8):
        for repeat in range(2):
            bank = adapter.extract(
                [sample["text"] for sample in samples],
                [sample["sample_id"] for sample in samples],
                batch_size=size,
            )
            save_artifact(output / f"batch-{size}-repeat-{repeat}.json", bank)
            banks.append(bank)
    comparison = compare_banks(banks, atol=atol, rtol=rtol)
    costs = [
        target["gradient_evaluations"]
        for bank in banks
        for sample in bank["samples"]
        for target in sample["attributions"]
    ]
    report = {
        "schema_version": 1,
        "role": "development",
        "development_sha256": identity,
        "runtime": adapter.runtime,
        "comparison": comparison,
        "bank_sha256": [sha256_json(bank) for bank in banks],
        "measured_cost": {
            "elapsed_extraction_seconds": sum(bank["latency_seconds"] for bank in banks),
            "actual_gradient_evaluations": sum(costs),
            "max_evaluations_per_target": max(costs),
            "source_examples": len(samples),
            "unique_texts": len({sample["text"] for sample in samples}),
            "fixed_targets_per_example": len(settings["candidate_ids"]) - 1,
            "executed_bank_passes": 6,
        },
        "G1": (
            "not_established: task schema/alignment/sign/permutation case coverage, "
            "all model/corpus signals and resource approvals are separate gates"
        ),
    }
    save_artifact(output / "qualification.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    report = qualify_reference(
        args.checkpoint,
        json.loads(args.development_json.read_text()),
        args.output,
        device=args.device,
        max_tokens=args.max_tokens,
        n_steps=args.n_steps,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(json.dumps(report, sort_keys=True))
    if not report["comparison"]["numerical_comparison_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
