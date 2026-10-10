"""Test the numerical IG rule and fixed token-map invariants without model weights."""

import numpy as np
import pytest
from pydantic import ValidationError

from expl_drift_endpoint.contract import AttributionPlan, AttributionSignal
from expl_drift_endpoint.laya_attribution import (
    MAX_IG_COMPUTE_STEPS,
    _validate_completeness,
    integrate_gradients,
    state_token_span,
)


def test_integrated_gradients_matches_linear_score_difference():
    inputs = np.array([[2.0, -1.0]])
    baseline = np.array([[0.0, 3.0]])
    weights = np.array([[4.0, -2.0]])

    attributions, steps = integrate_gradients(
        inputs,
        baseline,
        lambda _: weights,
        n_steps=16,
    )

    np.testing.assert_allclose(attributions, (inputs - baseline) * weights)
    expected_difference = (inputs * weights).sum() - (baseline * weights).sum()
    np.testing.assert_allclose(attributions.sum(), expected_difference)
    assert steps >= 16


def test_integrated_gradients_integrates_quadratic_score():
    inputs = np.array([[2.0, -3.0]])
    baseline = np.zeros_like(inputs)

    attributions, _ = integrate_gradients(
        inputs,
        baseline,
        lambda point: 2.0 * point,
        n_steps=16,
    )

    np.testing.assert_allclose(attributions, inputs**2)


def test_integrated_gradients_adaptively_integrates_cubic_gradient():
    inputs = np.array([[1.5]])
    baseline = np.zeros_like(inputs)

    attributions, _ = integrate_gradients(
        inputs,
        baseline,
        lambda point: 4.0 * point**3,
        n_steps=64,
        completeness_tolerance=1e-10,
    )

    np.testing.assert_allclose(attributions, inputs**4, rtol=1e-12, atol=1e-12)


def test_integrated_gradients_adaptively_integrates_smooth_path():
    inputs = np.array([[1.0]])
    baseline = np.zeros_like(inputs)

    attributions, steps = integrate_gradients(
        inputs,
        baseline,
        np.exp,
        n_steps=64,
        completeness_tolerance=1e-8,
    )

    np.testing.assert_allclose(attributions, np.expm1(inputs), rtol=1e-7, atol=1e-7)
    assert 64 <= steps <= MAX_IG_COMPUTE_STEPS


def test_completeness_tolerance_checks_absolute_and_relative_error():
    _validate_completeness(completeness_delta=0.05, logit_delta=4.0)
    _validate_completeness(completeness_delta=0.009, logit_delta=0.0)

    with pytest.raises(RuntimeError, match="exceeds tolerance"):
        _validate_completeness(completeness_delta=0.0501, logit_delta=4.0)
    with pytest.raises(RuntimeError, match="exceeds tolerance"):
        _validate_completeness(completeness_delta=0.0101, logit_delta=0.0)


def test_integrated_gradients_default_uses_64_steps_and_caps_requested_steps():
    plan = AttributionPlan(method="integrated_gradients", target_candidate_id="alpha")
    assert plan.n_steps == 64
    with pytest.raises(ValidationError, match="less than or equal to 64"):
        AttributionPlan(method="integrated_gradients", target_candidate_id="alpha", n_steps=65)


def test_integrated_gradients_rejects_invalid_steps_and_shapes():
    values = np.ones((1, 2))
    with pytest.raises(ValueError, match="n_steps"):
        integrate_gradients(values, values, lambda point: point, n_steps=1)
    with pytest.raises(ValueError, match="same shape"):
        integrate_gradients(values, np.ones((1, 3)), lambda point: point, n_steps=4)


def test_state_token_span_excludes_layas_closing_separator():
    assert state_token_span(12, {"state_tokens_used": 4}) == (7, 11)
    assert state_token_span(12, {"state_tokens_used": 0}) == (11, 11)
    with pytest.raises(ValueError, match="incompatible"):
        state_token_span(4, {"state_tokens_used": 4})


def test_attribution_contract_requires_aligned_fixed_width_arrays():
    payload = {
        "method": "integrated_gradients",
        "target_candidate_id": "alpha",
        "target_semantics": "candidate_logit",
        "baseline": "pad_document_tokens",
        "n_steps": 2048,
        "token_width": 3,
        "token_ids": [1, 2, 3],
        "tokens": ["[CLS]", "word", "[SEP]"],
        "segments": ["special", "document", "special"],
        "values": [0.0, 0.5, 0.0],
        "state_token_start": 1,
        "state_token_count": 1,
        "state_tokens_total": 2,
        "state_tokens_dropped": 1,
        "completeness_delta": 0.01,
    }
    assert AttributionSignal.model_validate(payload).token_width == 3
    payload["n_steps"] = 2049
    with pytest.raises(ValidationError, match="less than or equal to 2048"):
        AttributionSignal.model_validate(payload)

    payload["n_steps"] = 2048

    payload["values"] = [0.0, 0.5]
    with pytest.raises(ValidationError, match="must match token_width"):
        AttributionSignal.model_validate(payload)
