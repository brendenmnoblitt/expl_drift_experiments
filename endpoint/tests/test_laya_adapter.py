"""CPU autograd fixtures qualify declared functions, not deployed checkpoints."""

import copy

import numpy as np
import pytest

from expl_drift_endpoint.contract import ExtractionRequest
from expl_drift_endpoint.laya_adapter import IncompatibleRevision, LayaAdapter


def test_fixed_signed_margin_and_document_pool_share_forward_function(
    request_payload,
    runtime_identity,
    fake_agent,
):
    adapter = LayaAdapter(fake_agent, "/fixture", runtime=runtime_identity)
    response = adapter.extract(ExtractionRequest.model_validate(request_payload))
    positive, negative = response.samples
    assert positive.target_value == pytest.approx(8.0)
    assert negative.target_value == pytest.approx(-16.0)
    assert positive.predicted_candidate_id == "beta"
    assert negative.predicted_candidate_id == "alpha"
    for sample, expected_values, expected_pool in (
        (positive, [16.0, -8.0], [1.0, 0.5]),
        (negative, [-8.0, -8.0], [-2.0, -1.0]),
    ):
        signal = sample.attribution
        assert signal.target_candidate_id == "beta"
        assert signal.reference_candidate_id == "alpha"
        assert signal.baseline_target_value == 0.0
        np.testing.assert_allclose(signal.values[3:5], expected_values, atol=1e-6)
        assert sum(signal.values) == pytest.approx(sample.target_value, abs=1e-6)
        np.testing.assert_allclose(sample.representation.values, expected_pool)
        assert sample.tokens.state_tokens_dropped == 1
        logits = np.array([score.logit for score in sample.scores])
        expected_probs = np.exp((logits - logits.max()) / sample.probability_temperature)
        expected_probs /= expected_probs.sum()
        np.testing.assert_allclose(
            [score.probability for score in sample.scores], expected_probs, atol=1e-7
        )


@pytest.mark.parametrize("field", ["model", "runtime"])
def test_different_checkpoint_or_runtime_cannot_execute(
    request_payload, runtime_identity, fake_agent, field
):
    payload = copy.deepcopy(request_payload)
    if field == "model":
        payload[field]["revision"] = "c" * 40
    else:
        payload[field]["implementation_sha256"] = "c" * 64
    adapter = LayaAdapter(fake_agent, "/fixture", runtime=runtime_identity)
    with pytest.raises(IncompatibleRevision):
        adapter.extract(ExtractionRequest.model_validate(payload))
