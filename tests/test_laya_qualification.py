import json
import math

import pytest

from experiments.research_extension.laya_qualification import (
    MAX_GRADIENT_EVALUATIONS_PER_TARGET,
    MAX_REQUESTS_PER_TARGET,
    MODEL,
    TASKS,
    qualify_laya,
)

IMAGE = "sha256:" + "a" * 64


def development(task_id):
    candidate_ids = [candidate["candidate_id"] for candidate in TASKS[task_id]["candidates"]]
    return {
        "role": "development",
        "task": {
            "candidate_ids": candidate_ids,
            "reference_candidate_id": candidate_ids[0],
        },
        "samples": [{"sample_id": f"{task_id}-{i}", "text": f"sample {i}"} for i in range(8)],
    }


class FakeClient:
    fail_at: int | None

    def __init__(self, base_url, token, *, timeout):
        self.base_url = base_url
        self.token = token
        self.timeout = timeout
        self.calls = []
        self.fail_at = None

    def health(self):
        return {
            "schema_version": "3",
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
        if len(self.calls) == self.fail_at:
            raise TimeoutError("simulated bounded request timeout")
        candidate_ids = [candidate["candidate_id"] for candidate in request["candidates"]]
        logits = [1.0 - index * 0.25 for index in range(len(candidate_ids))]
        exponentials = [math.exp(value - max(logits)) for value in logits]
        probabilities = [value / sum(exponentials) for value in exponentials]
        target = request["decision_target"]["target_candidate_id"]
        reference = request["decision_target"]["reference_candidate_id"]
        margin = logits[candidate_ids.index(target)] - logits[candidate_ids.index(reference)]
        rows = []
        for item in request["samples"]:
            tokens = {
                "token_width": 1,
                "token_ids": [11],
                "tokens": ["sample"],
                "segments": ["document"],
                "state_token_start": 0,
                "state_token_count": 1,
                "state_tokens_total": 1,
                "state_tokens_dropped": 0,
            }
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "predicted_candidate_id": candidate_ids[0],
                    "scores": [
                        {"candidate_id": candidate, "logit": logit, "probability": probability}
                        for candidate, logit, probability in zip(
                            candidate_ids, logits, probabilities, strict=True
                        )
                    ],
                    "tokens": tokens,
                    "representation": {"values": [0.25, -0.5]},
                    "attribution": {
                        "values": [margin],
                        "input_target_value": margin,
                        "baseline_target_value": 0.0,
                        "completeness_delta": 0.0,
                        "gradient_evaluations": 64,
                    },
                }
            )
        return {
            "decision_target": request["decision_target"],
            "samples": rows,
            "latency_seconds": 0.01,
        }


def run(tmp_path, task_id, client, **overrides):
    return qualify_laya(
        development(task_id),
        TASKS[task_id],
        task_id,
        tmp_path / task_id,
        endpoint_url="https://example.invalid",
        token="not-a-real-token",
        image_digest=IMAGE,
        allow_remote_execution=True,
        client_factory=lambda *args, **kwargs: client,
        **overrides,
    )


def test_remote_execution_requires_explicit_approval_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="explicit approval"):
        qualify_laya(
            development("youtube"),
            TASKS["youtube"],
            "youtube",
            tmp_path / "remote",
            endpoint_url="https://example.invalid",
            token="token",
            image_digest=IMAGE,
        )
    assert not (tmp_path / "remote").exists()


@pytest.mark.parametrize(
    ("task_id", "expected_targets", "expected_requests", "gradient_budget"),
    [
        ("youtube", 1, 22, 786432),
        ("ag_news", 3, 66, 2359296),
    ],
)
def test_tasks_qualify_all_fixed_margins_with_bounded_persisted_requests(
    task_id, expected_targets, expected_requests, gradient_budget, tmp_path
):
    client = FakeClient("url", "token", timeout=1)
    report = run(tmp_path, task_id, client)

    assert len(client.calls) == expected_requests
    assert expected_requests == MAX_REQUESTS_PER_TARGET * expected_targets
    assert (
        len({request["decision_target"]["target_candidate_id"] for request in client.calls})
        == expected_targets
    )
    assert [len(request["samples"]) for request in client.calls].count(1) == 16 * expected_targets
    assert [len(request["samples"]) for request in client.calls].count(4) == 4 * expected_targets
    assert [len(request["samples"]) for request in client.calls].count(8) == 2 * expected_targets
    output = tmp_path / task_id
    assert len(list((output / "responses").glob("response-*.json"))) == expected_requests
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert len(events) == expected_requests
    assert all(event["status"] == "completed" for event in events)
    assert report["comparison"]["numerical_comparison_passed"]
    assert report["measured_cost"]["requests_completed"] == expected_requests
    plan = json.loads((output / "comparison-plan.json").read_text())
    assert plan["max_total_gradient_evaluations"] == gradient_budget
    assert plan["max_gradient_evaluations_per_target"] == MAX_GRADIENT_EVALUATIONS_PER_TARGET
    assert plan["max_request_timeout_seconds"] == 900
    assert plan["max_total_wall_seconds"] == 7200
    assert report["G1"].startswith("not_established:")


def test_failed_request_is_logged_without_retry_and_prior_responses_remain(tmp_path):
    client = FakeClient("url", "token", timeout=1)
    client.fail_at = 3
    with pytest.raises(TimeoutError, match="simulated"):
        run(tmp_path, "youtube", client)

    output = tmp_path / "youtube"
    assert len(client.calls) == 3
    assert len(list((output / "responses").glob("response-*.json"))) == 2
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert [event["status"] for event in events] == ["completed", "completed", "failed"]
    assert (output / "run-failure.json").exists()


@pytest.mark.parametrize(
    "role", ["training", "reference", "held_out_stable", "held_out_evaluation"]
)
def test_non_development_roles_are_refused_without_output(role, tmp_path):
    data = {**development("youtube"), "role": role}
    with pytest.raises(ValueError, match="designated development"):
        qualify_laya(
            data,
            TASKS["youtube"],
            "youtube",
            tmp_path / "remote",
            endpoint_url="url",
            token="token",
            image_digest=IMAGE,
            allow_remote_execution=True,
        )
    assert not (tmp_path / "remote").exists()
