"""Exercise the HTTP boundary without model downloads or legacy package imports."""

import copy
import json

import pytest
from fastapi.testclient import TestClient

from expl_drift_endpoint.app import create_app
from expl_drift_endpoint.contract import ExtractionRequest


@pytest.fixture
def client():
    with TestClient(create_app()) as value:
        yield value


@pytest.fixture
def payload():
    # Revisions here are synthetic contract fixtures, not deployable checkpoints.
    revision = {"repository": "fixture/model", "revision": "a" * 40}
    return {
        "schema_version": "1",
        "request_id": "request-1",
        "run_id": "run-1",
        "window_index": 0,
        "corpus_sha256": "b" * 64,
        "model": revision,
        "tokenizer": revision,
        "instructions": "Choose the department that should handle the ticket.",
        "candidates": [
            {"candidate_id": "billing", "description": "Invoices, charges, refunds"},
            {"candidate_id": "technical", "description": "Bugs and outages"},
        ],
        "samples": [{"sample_id": "ticket-1", "text": "  I was charged twice.\n"}],
        "max_tokens": 512,
        "attribution": {"method": "integrated_gradients", "target_candidate_id": "billing"},
    }


def test_liveness_does_not_claim_model_readiness(client):
    assert client.get("/live").json() == {"status": "alive"}
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["extraction_available"] is False


def test_valid_request_is_unavailable_and_returns_no_signals(client, payload):
    response = client.post("/extract", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"]["request_id"] == "request-1"
    assert "signals" not in response.json()


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", "2"), ("window_index", True), ("max_tokens", "512")],
)
def test_incompatible_schema_or_scalar_coercion_is_rejected(client, payload, field, value):
    payload[field] = value
    assert client.post("/extract", json=payload).status_code == 422


def test_unknown_fields_are_rejected_at_both_levels(client, payload):
    payload["secret_typo"] = "unexpected"
    assert client.post("/extract", json=payload).status_code == 422
    del payload["secret_typo"]
    payload["samples"][0]["extra"] = "unexpected"
    assert client.post("/extract", json=payload).status_code == 422


def test_mutable_model_revision_is_rejected(client, payload):
    payload["model"] = {"repository": "fixture/model", "revision": "main"}
    assert client.post("/extract", json=payload).status_code == 422


@pytest.mark.parametrize("field", ["samples", "candidates"])
def test_duplicate_identities_are_rejected(client, payload, field):
    payload[field].append(copy.deepcopy(payload[field][0]))
    assert client.post("/extract", json=payload).status_code == 422


def test_unknown_attribution_target_is_rejected(client, payload):
    payload["attribution"]["target_candidate_id"] = "missing"
    assert client.post("/extract", json=payload).status_code == 422


@pytest.mark.parametrize("field", ["samples", "candidates"])
def test_empty_batch_or_candidate_set_is_rejected(client, payload, field):
    payload[field] = []
    assert client.post("/extract", json=payload).status_code == 422


def test_contract_preserves_text_and_candidate_order(payload):
    request = ExtractionRequest.model_validate_json(json.dumps(payload))
    assert request.samples[0].text == "  I was charged twice.\n"
    assert [candidate.candidate_id for candidate in request.candidates] == ["billing", "technical"]
