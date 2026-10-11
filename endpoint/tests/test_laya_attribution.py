"""Test the numerical IG rule and fixed token-map invariants without model weights."""

import numpy as np
import pytest
from pydantic import ValidationError

from expl_drift_endpoint.contract import SampleScores
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
    assert steps >= 2


def test_integrated_gradients_returns_a_bounded_estimate_when_budget_is_small():
    inputs = np.array([[1.0]])
    baselines = np.zeros_like(inputs)

    attributions, steps = integrate_gradients(
        inputs,
        baselines,
        lambda _: np.ones_like(inputs),
        n_steps=64,
        max_steps=2,
    )

    np.testing.assert_allclose(attributions, inputs)
    assert steps == 2


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


def test_integrated_gradients_integrates_degree_31_path_with_composite_rule():
    inputs = np.ones((1, 1))
    baseline = np.zeros_like(inputs)

    attributions, steps = integrate_gradients(
        inputs,
        baseline,
        lambda point: point**31,
        n_steps=64,
    )

    np.testing.assert_allclose(attributions, [[1.0 / 32.0]], rtol=1e-12, atol=1e-14)
    assert steps == 64


def test_adaptive_integrated_gradients_refines_the_interval_with_a_kink():
    inputs = np.ones((1, 1))
    baseline = np.zeros_like(inputs)

    attributions, steps = integrate_gradients(
        inputs,
        baseline,
        lambda point: np.where(point < 0.503, 1.0, 2.0),
        n_steps=64,
        max_steps=2048,
        target_delta=1.497,
        completeness_tolerance=1e-4,
    )

    np.testing.assert_allclose(attributions, [[1.497]], atol=1e-4)
    assert 96 < steps <= 2048


def test_integrated_gradients_integrates_smooth_path():
    inputs = np.array([[1.0]])
    baseline = np.zeros_like(inputs)

    attributions, steps = integrate_gradients(
        inputs,
        baseline,
        np.exp,
        n_steps=64,
    )

    np.testing.assert_allclose(attributions, np.expm1(inputs), rtol=1e-12, atol=1e-12)
    assert 2 <= steps <= MAX_IG_COMPUTE_STEPS


def test_completeness_tolerance_checks_absolute_and_relative_error():
    _validate_completeness(completeness_delta=0.05, logit_delta=4.0)
    _validate_completeness(completeness_delta=0.009, logit_delta=0.0)

    with pytest.raises(RuntimeError, match="exceeds tolerance"):
        _validate_completeness(completeness_delta=0.0501, logit_delta=4.0)
    with pytest.raises(RuntimeError, match="exceeds tolerance"):
        _validate_completeness(completeness_delta=0.0101, logit_delta=0.0)


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


@pytest.mark.parametrize("change", ["token_width", "document_span", "baseline_quantity"])
def test_saved_attribution_contract_rejects_misalignment_and_unrecalculable_residual(
    response_payload,
    change,
):
    payload = response_payload["samples"][0]
    if change == "token_width":
        payload["attribution"]["values"].pop()
    elif change == "document_span":
        payload["tokens"]["state_token_start"] += 1
    else:
        payload["attribution"]["baseline_target_value"] += 1.0
    with pytest.raises(ValidationError):
        SampleScores.model_validate(payload)
