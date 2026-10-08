"""Check stable request-to-Laya score mapping without loading model weights."""

import copy

import pytest
from fastapi.testclient import TestClient

from expl_drift_endpoint.app import create_app
from expl_drift_endpoint.contract import ExtractionRequest
from expl_drift_endpoint.laya_adapter import (
    MODEL_ID,
    MODEL_REVISION,
    QUESTION_ID,
    IncompatibleRevision,
    LayaAdapter,
    UnsupportedCapability,
)


class FakeAgent:
    cfg = {"max_len": 512}

    def predict_batch(self, states, questions, *, batch_size, max_len):
        assert states == ["first document", "second document"]
        assert batch_size == 2
        assert max_len == 128
        assert list(questions[QUESTION_ID]["criteria"]) == ["alpha", "beta"]
        return [
            {
                "answers": {
                    QUESTION_ID: {
                        "choice": "beta",
                        "probabilities": {"beta": 0.75, "alpha": 0.25},
                    }
                }
            },
            {
                "answers": {
                    QUESTION_ID: {
                        "choice": "alpha",
                        "probabilities": {"alpha": 0.875, "beta": 0.125},
                    }
                }
            },
        ]


def request_payload():
    revision = {"repository": MODEL_ID, "revision": MODEL_REVISION}
    return {
        "schema_version": "1",
        "request_id": "request-1",
        "run_id": "run-1",
        "window_index": 3,
        "corpus_sha256": "b" * 64,
        "model": revision,
        "tokenizer": revision,
        "instructions": "Choose the matching category.",
        "candidates": [
            {"candidate_id": "alpha", "description": "First category"},
            {"candidate_id": "beta", "description": "Second category"},
        ],
        "samples": [
            {"sample_id": "sample-1", "text": "first document"},
            {"sample_id": "sample-2", "text": "second document"},
        ],
        "max_tokens": 128,
        "attribution": None,
        "representations": False,
    }


def make_adapter():
    return LayaAdapter(FakeAgent(), model_path="/fixture/laya")


def test_adapter_preserves_sample_and_candidate_order_and_probabilities():
    request = ExtractionRequest.model_validate(request_payload())

    response = make_adapter().extract(request)

    assert [sample.sample_id for sample in response.samples] == ["sample-1", "sample-2"]
    assert [score.candidate_id for score in response.samples[0].scores] == ["alpha", "beta"]
    assert [score.probability for score in response.samples[0].scores] == [0.25, 0.75]
    assert response.samples[0].predicted_candidate_id == "beta"
    assert response.samples[1].predicted_candidate_id == "alpha"
    assert response.score_semantics == "choice_probability"


def test_adapter_rejects_unqualified_attribution_and_wrong_revision():
    data = request_payload()
    data["attribution"] = {"method": "attention", "target_candidate_id": "alpha"}
    request = ExtractionRequest.model_validate(data)
    with pytest.raises(
        UnsupportedCapability,
        match="implemented Laya attribution method is integrated_gradients",
    ):
        make_adapter().extract(request)

    data = copy.deepcopy(request_payload())
    data["model"] = {"repository": MODEL_ID, "revision": "c" * 40}
    request = ExtractionRequest.model_validate(data)
    with pytest.raises(IncompatibleRevision):
        make_adapter().extract(request)


def test_endpoint_reports_loaded_score_capability_and_serializes_results():
    payload = request_payload()
    response_adapter = make_adapter()
    with TestClient(create_app(adapter_factory=lambda: response_adapter)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["capabilities"] == ["decision_scores"]
        response = client.post("/extract", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert [sample["sample_id"] for sample in body["samples"]] == ["sample-1", "sample-2"]
    assert body["samples"][0]["scores"] == [
        {"candidate_id": "alpha", "probability": 0.25},
        {"candidate_id": "beta", "probability": 0.75},
    ]
