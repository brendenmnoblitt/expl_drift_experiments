"""Exercise strict HTTP errors and an actual autograd extraction response."""

import copy

import pytest
from fastapi.testclient import TestClient

from expl_drift_endpoint.app import create_app
from expl_drift_endpoint.laya_adapter import LayaAdapter


@pytest.fixture
def client():
    with TestClient(create_app()) as value:
        yield value


def test_liveness_does_not_claim_model_readiness(client):
    assert client.get("/live").json() == {"status": "alive"}
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["extraction_available"] is False


def test_unloaded_service_returns_unavailable_not_fake_signals(client, request_payload):
    response = client.post("/extract", json=request_payload)
    assert response.status_code == 503
    assert "samples" not in response.json()


@pytest.mark.parametrize(
    "field,value", [("schema_version", "1"), ("window_index", True), ("max_tokens", "8")]
)
def test_incompatible_schema_or_scalar_coercion_is_rejected(client, request_payload, field, value):
    request_payload[field] = value
    assert client.post("/extract", json=request_payload).status_code == 422


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(secret_typo="unexpected"),
        lambda p: p["samples"][0].update(extra="unexpected"),
        lambda p: p["model"].update(revision="main"),
        lambda p: p["decision_target"].update(target_candidate_id="missing"),
        lambda p: p["decision_target"].update(target_candidate_id="alpha"),
        lambda p: p.update(samples=[]),
        lambda p: p.update(candidates=[]),
        lambda p: p["samples"].append(copy.deepcopy(p["samples"][0])),
        lambda p: p["candidates"].append(copy.deepcopy(p["candidates"][0])),
    ],
)
def test_invalid_task_and_identity_are_rejected(client, request_payload, mutate):
    mutate(request_payload)
    assert client.post("/extract", json=request_payload).status_code == 422


def test_http_extraction_keeps_mount_diagnostics_out_of_semantic_identity(
    request_payload, runtime_identity, checkpoint_mount, fake_agent
):
    adapter = LayaAdapter(
        fake_agent, "/fixture", runtime=runtime_identity, checkpoint_mount=checkpoint_mount
    )
    with TestClient(create_app(adapter_factory=lambda: adapter)) as client:
        health = client.get("/health").json()
        assert health["checkpoint_mount"] == checkpoint_mount.model_dump()
        assert "checkpoint_artifact_sha256" not in health["runtime"]
        response = client.post("/extract", json=request_payload)
        assert response.status_code == 200
        assert response.json()["runtime"] == runtime_identity.model_dump()
        assert "checkpoint_files" not in response.json()["runtime"]
        changed = copy.deepcopy(request_payload)
        changed["runtime"]["configured_image_digest"] = "sha256:" + "c" * 64
        assert client.post("/extract", json=changed).status_code == 409
