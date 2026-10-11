"""Validate remote Laya responses before the experiment consumes them."""

import copy
import math
import sys
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
client = import_module("experiments.laya_endpoint.client")
validate_response = client.validate_response


@pytest.fixture
def request_payload():
    revision = {"repository": "convaiinnovations/laya", "revision": "a" * 40}
    return {
        "schema_version": "1",
        "request_id": "request-1",
        "run_id": "run-1",
        "window_index": 0,
        "model": revision,
        "tokenizer": revision,
        "candidates": [
            {"candidate_id": "ham", "description": "A genuine comment."},
            {"candidate_id": "spam", "description": "An unsolicited promotional comment."},
        ],
        "samples": [{"sample_id": "youtube-1", "text": "A short comment."}],
        "attribution": {
            "method": "integrated_gradients",
            "target_candidate_id": "spam",
            "n_steps": 64,
        },
    }


@pytest.fixture
def response_payload(request_payload):
    revision = request_payload["model"]
    return {
        "schema_version": "1",
        "request_id": request_payload["request_id"],
        "run_id": request_payload["run_id"],
        "window_index": request_payload["window_index"],
        "model": revision,
        "tokenizer": revision,
        "score_semantics": "choice_probability",
        "samples": [
            {
                "sample_id": "youtube-1",
                "predicted_candidate_id": "spam",
                "scores": [
                    {"candidate_id": "ham", "probability": 0.3},
                    {"candidate_id": "spam", "probability": 0.7},
                ],
                "attribution": {
                    "method": "integrated_gradients",
                    "target_candidate_id": "spam",
                    "target_semantics": "candidate_logit",
                    "baseline": "pad_document_tokens",
                    "n_steps": 64,
                    "token_width": 3,
                    "token_ids": [101, 202, 102],
                    "tokens": ["[CLS]", "comment", "[SEP]"],
                    "segments": ["special", "document", "special"],
                    "values": [0.0, 0.5, 0.0],
                    "state_token_start": 1,
                    "state_token_count": 1,
                    "state_tokens_total": 1,
                    "state_tokens_dropped": 0,
                    "completeness_delta": 0.001,
                },
            }
        ],
    }


def test_accepts_maximum_adaptive_ig_evaluation_count(request_payload, response_payload):
    response = copy.deepcopy(response_payload)
    response["samples"][0]["attribution"]["n_steps"] = 16384

    validate_response(request_payload, response)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda response: response["samples"][0]["scores"].reverse(), "score order"),
        (
            lambda response: response["samples"][0]["scores"][1].update(probability=0.2),
            "do not sum to one",
        ),
        (
            lambda response: response["samples"][0]["attribution"]["values"].append(math.nan),
            "misaligned",
        ),
        (
            lambda response: response["samples"][0]["attribution"].update(state_tokens_dropped=1),
            "truncation count",
        ),
        (
            lambda response: response["samples"][0]["attribution"].update(completeness_delta=0.1),
            "completeness residual",
        ),
        (
            lambda response: response["samples"][0]["attribution"].update(n_steps=16385),
            "evaluation count",
        ),
    ],
)
def test_rejects_invalid_scores_or_ig_alignment(request_payload, response_payload, mutate, message):
    response = copy.deepcopy(response_payload)
    mutate(response)

    with pytest.raises(ValueError, match=message):
        validate_response(request_payload, response)


def test_split_batch_cache_resumes_and_merges_validated_subresponses(
    tmp_path, request_payload, response_payload
):
    request = copy.deepcopy(request_payload)
    request["samples"] = [
        {"sample_id": f"youtube-{index}", "text": f"Comment {index}."} for index in range(9)
    ]
    request["corpus_sha256"] = client.sha256_json(request["samples"])
    subrequests = client.split_request(request)

    assert [len(part["samples"]) for part in subrequests] == [8, 1]
    assert [sample for part in subrequests for sample in part["samples"]] == request["samples"]

    def part_record(part):
        response = copy.deepcopy(response_payload)
        response["request_id"] = part["request_id"]
        response["run_id"] = part["run_id"]
        response["window_index"] = part["window_index"]
        response["samples"] = []
        for sample in part["samples"]:
            output = copy.deepcopy(response_payload["samples"][0])
            output["sample_id"] = sample["sample_id"]
            response["samples"].append(output)
        return {
            "request_sha256": client.sha256_json(part),
            "request": part,
            "endpoint_image_digest": "sha256:" + "a" * 64,
            "response": response,
        }

    cache_path = tmp_path / "window.json"
    cache_entry = {
        "request_sha256": client.sha256_json(request),
        "subresponses": [part_record(subrequests[0])],
    }
    client.atomic_write_json(cache_path, cache_entry)
    assert client.load_cached_response(cache_path, request) is None

    cache_entry["subresponses"].append(part_record(subrequests[1]))
    client.atomic_write_json(cache_path, cache_entry)
    merged = client.load_cached_response(cache_path, request)

    assert [sample["sample_id"] for sample in merged["samples"]] == [
        sample["sample_id"] for sample in request["samples"]
    ]
    assert merged["request_id"] == request["request_id"]
