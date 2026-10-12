import json

import pytest

from experiments.research_extension.laya_qualification import (
    MAX_REQUESTS,
    MAX_TOTAL_GRADIENT_EVALUATIONS,
    MODEL,
    qualify_laya,
)

IMAGE = "sha256:" + "a" * 64
CANDIDATES = [
    {"candidate_id": "ham", "description": "A genuine comment."},
    {"candidate_id": "spam", "description": "A spam comment."},
]
DEVELOPMENT = {
    "role": "development",
    "task": {"candidate_ids": ["ham", "spam"], "reference_candidate_id": "ham"},
    "samples": [{"sample_id": f"dev-{i}", "text": f"comment {i}"} for i in range(8)],
}
TASK = {"instructions": "Classify this comment.", "candidates": CANDIDATES}


class FakeClient:
    def __init__(self, base_url, token, *, timeout):
        self.base_url = base_url
        self.calls = []
        self.fail_at = None

    def health(self):
        return {
            "schema_version": "2",
            "model": MODEL,
            "runtime": {"configured_image_digest": IMAGE},
            "capabilities": [
                "integrated_gradients",
                "fixed_candidate_margin",
                "document_mean_representation",
            ],
        }

    def extract(self, request):
        self.calls.append(request)
        if len(self.calls) == getattr(self, "fail_at", None):
            raise TimeoutError("simulated bounded request timeout")
        rows = []
        for item in request["samples"]:
            tokens = {
                "token_width": 1,
                "token_ids": [11],
                "tokens": ["comment"],
                "segments": ["document"],
                "state_token_start": 0,
                "state_token_count": 1,
                "state_tokens_total": 1,
                "state_tokens_dropped": 0,
            }
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "predicted_candidate_id": "ham",
                    "scores": [
                        {"candidate_id": "ham", "logit": 1.0, "probability": 0.7310586},
                        {"candidate_id": "spam", "logit": 0.0, "probability": 0.2689414},
                    ],
                    "tokens": tokens,
                    "representation": {"values": [0.25, -0.5]},
                    "attribution": {
                        "values": [-1.0],
                        "input_target_value": -1.0,
                        "baseline_target_value": 0.0,
                        "completeness_delta": 0.0,
                        "gradient_evaluations": 64,
                    },
                }
            )
        return {"samples": rows, "latency_seconds": 0.01}


def run(tmp_path, client, **overrides):
    return qualify_laya(
        DEVELOPMENT,
        TASK,
        tmp_path / "remote",
        endpoint_url="https://example.invalid",
        token="not-a-real-token",
        image_digest=IMAGE,
        allow_remote_execution=True,
        client_factory=lambda *args, **kwargs: client,
        **overrides,
    )


def test_remote_execution_requires_explicit_approval_before_creating_output(tmp_path):
    output = tmp_path / "remote"
    with pytest.raises(ValueError, match="explicit approval"):
        qualify_laya(
            DEVELOPMENT,
            TASK,
            output,
            endpoint_url="https://example.invalid",
            token="token",
            image_digest=IMAGE,
        )
    assert not output.exists()


def test_success_is_bounded_and_persists_each_remote_result(tmp_path):
    client = FakeClient("url", "token", timeout=1)
    report = run(tmp_path, client)

    assert len(client.calls) == MAX_REQUESTS == 22
    assert [len(request["samples"]) for request in client.calls].count(1) == 16
    assert [len(request["samples"]) for request in client.calls].count(4) == 4
    assert [len(request["samples"]) for request in client.calls].count(8) == 2
    assert all(
        request["decision_target"]["target_candidate_id"] == "spam" for request in client.calls
    )
    output = tmp_path / "remote"
    assert len(list((output / "responses").glob("response-*.json"))) == MAX_REQUESTS
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert len(events) == MAX_REQUESTS
    assert all(event["status"] == "completed" for event in events)
    assert report["comparison"]["numerical_comparison_passed"]
    plan = json.loads((output / "comparison-plan.json").read_text())
    assert plan["max_total_gradient_evaluations"] == MAX_TOTAL_GRADIENT_EVALUATIONS == 786432
    assert plan["max_request_timeout_seconds"] == 900
    assert plan["max_total_wall_seconds"] == 7200
    assert report["measured_cost"]["requests_completed"] == MAX_REQUESTS
    assert report["G1"].startswith("not_established:")


def test_failed_request_is_logged_without_retry_and_prior_responses_remain(tmp_path):
    client = FakeClient("url", "token", timeout=1)
    client.fail_at = 3
    with pytest.raises(TimeoutError, match="simulated"):
        run(tmp_path, client)

    output = tmp_path / "remote"
    assert len(client.calls) == 3
    assert len(list((output / "responses").glob("response-*.json"))) == 2
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert [event["status"] for event in events] == ["completed", "completed", "failed"]
    assert (output / "run-failure.json").exists()


@pytest.mark.parametrize(
    "role", ["training", "reference", "held_out_stable", "held_out_evaluation"]
)
def test_non_development_roles_are_refused_without_output(role, tmp_path):
    development = {**DEVELOPMENT, "role": role}
    with pytest.raises(ValueError, match="designated development"):
        qualify_laya(
            development,
            TASK,
            tmp_path / "remote",
            endpoint_url="url",
            token="token",
            image_digest=IMAGE,
            allow_remote_execution=True,
        )
    assert not (tmp_path / "remote").exists()
