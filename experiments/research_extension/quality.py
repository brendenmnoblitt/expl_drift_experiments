"""Empirical classification quality and composition-only expectations from saved logits.

These are point estimates, not group-aware intervals or qualified quality events.
Raw logits and an explicit temperature keep log loss valid when probabilities
underflow. Label-only controls need no model or attribution calls.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def _weighted_recall(
    prior: np.ndarray, recalls: Sequence[float | None]
) -> tuple[float | None, list[int]]:
    unknown = [i for i, value in enumerate(recalls) if value is None and prior[i] > 0]
    if unknown:
        return None, unknown
    return math.fsum(
        float(prior[i]) * value for i, value in enumerate(recalls) if value is not None
    ), []


def classification_quality(
    logits: Any,
    labels: Any,
    *,
    candidate_ids: Sequence[str],
    temperature: float = 1.0,
    standardized_prior: Sequence[float] | None = None,
    reference_recalls: Sequence[float | None] | None = None,
) -> dict[str, Any]:
    """Report ordered-task quality; reference recalls must belong to the same fixed model.

    Balanced accuracy averages classes present in this batch. Standardized-prior
    accuracy instead uses the declared population prior (uniform by default),
    and is undefined when that prior needs an unobserved class. Composition-only
    accuracy weights reference class recalls by this batch's observed prior;
    it does not attribute within-class performance changes to prior shift.
    """
    candidates = tuple(candidate_ids)
    if (
        len(candidates) < 2
        or len(set(candidates)) != len(candidates)
        or any(not isinstance(candidate, str) or not candidate for candidate in candidates)
    ):
        raise ValueError("candidate IDs must be nonempty, ordered and unique")
    scores = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels)
    if (
        scores.ndim != 2
        or scores.shape[1] != len(candidates)
        or scores.shape[0] == 0
        or not np.isfinite(scores).all()
        or truth.shape != (scores.shape[0],)
        or truth.dtype.kind not in "iu"
        or (truth < 0).any()
        or (truth >= len(candidates)).any()
    ):
        raise ValueError("finite logits and aligned dense integer labels are required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    prior = np.asarray(
        standardized_prior
        if standardized_prior is not None
        else np.full(len(candidates), 1.0 / len(candidates)),
        dtype=np.float64,
    )
    if (
        prior.shape != (len(candidates),)
        or not np.isfinite(prior).all()
        or (prior < 0).any()
        or not math.isclose(float(prior.sum()), 1.0, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("standardized prior must be a declared probability distribution")
    with np.errstate(over="raise", invalid="raise"):
        scaled = scores / temperature
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probabilities)
    predictions = scores.argmax(axis=1)
    counts = np.bincount(truth.astype(np.int64), minlength=len(candidates))
    recalls = [
        float((predictions[truth == i] == i).mean()) if count else None
        for i, count in enumerate(counts)
    ]
    present = [value for value in recalls if value is not None]
    observed_prior = counts / len(truth)
    standardized, unknown_standardized = _weighted_recall(prior, recalls)
    composition = None
    unknown_reference = []
    if reference_recalls is not None:
        if len(reference_recalls) != len(candidates) or any(
            value is not None and (not math.isfinite(value) or not 0 <= value <= 1)
            for value in reference_recalls
        ):
            raise ValueError("reference recalls must match the task and lie in [0, 1] or be None")
        composition, unknown_reference = _weighted_recall(observed_prior, reference_recalls)
    errors = probabilities.copy()
    errors[np.arange(len(truth)), truth] -= 1.0
    return {
        "candidate_ids": list(candidates),
        "count": len(truth),
        "class_counts": {str(i): int(count) for i, count in enumerate(counts)},
        "class_prior": observed_prior.tolist(),
        "accuracy": float((predictions == truth).mean()),
        "balanced_accuracy": math.fsum(present) / len(present),
        "balanced_accuracy_scope": "classes_present_in_this_batch",
        "per_class_recall": {str(i): value for i, value in enumerate(recalls)},
        "standardized_prior": prior.tolist(),
        "standardized_prior_accuracy": standardized,
        "undefined_standardized_classes": unknown_standardized,
        "log_loss": float(-log_probabilities[np.arange(len(truth)), truth].mean()),
        "multiclass_brier": float(np.square(errors).sum(axis=1).mean()),
        "composition_only_expected_accuracy": composition,
        "composition_reference_supplied": reference_recalls is not None,
        "undefined_composition_reference_classes": unknown_reference,
        "temperature": temperature,
        "analysis_precision": "float64",
    }
