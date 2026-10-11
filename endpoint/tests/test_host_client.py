"""Reject incompatible numerical signals and exact execution/cache identities."""

import copy
import math

import pytest
from experiments.laya_endpoint import client

from expl_drift_endpoint.contract import ExtractionRequest
from expl_drift_endpoint.laya_adapter import LayaAdapter


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["samples"][0]["scores"].reverse(),
        lambda r: r["samples"][0]["scores"][0].update(logit=999.0),
        lambda r: r["samples"][0]["attribution"].update(baseline_target_value=999.0),
        lambda r: r["samples"][0]["attribution"].update(completeness_delta=0.5),
        lambda r: r["samples"][0]["attribution"].update(gradient_evaluations=16385),
        lambda r: r["samples"][0]["attribution"]["values"].append(math.nan),
        lambda r: r["samples"][0]["tokens"].update(state_tokens_dropped=0),
        lambda r: r["samples"][0]["representation"].update(values=[math.nan]),
    ],
)
def test_changed_scores_or_incomplete_quantities_are_rejected(
    request_payload, response_payload, mutate
):
    response = copy.deepcopy(response_payload)
    mutate(response)
    with pytest.raises(ValueError):
        client.validate_response(request_payload, response)


@pytest.mark.parametrize("change", ["text", "target", "precision", "runtime", "batch_order"])
def test_cache_is_bound_to_complete_fingerprint(
    tmp_path, request_payload, response_payload, change
):
    path = tmp_path / "cache.json"
    client.atomic_write_json(
        path, {"request_sha256": client.sha256_json(request_payload), "response": response_payload}
    )
    changed = copy.deepcopy(request_payload)
    if change == "text":
        changed["samples"][0]["text"] += "!"
    elif change == "target":
        (
            changed["decision_target"]["target_candidate_id"],
            changed["decision_target"]["reference_candidate_id"],
        ) = "alpha", "beta"
    elif change == "precision":
        changed["precision"] = "other_path"
    elif change == "runtime":
        changed["runtime"]["configured_image_digest"] = "sha256:" + "c" * 64
    else:
        changed["samples"].reverse()
    with pytest.raises(ValueError):
        client.load_cached_response(path, changed)


def test_partial_cache_replays_exact_children_and_keeps_actual_batch_context(
    tmp_path,
    request_payload,
    runtime_identity,
    fake_agent,
):
    request = copy.deepcopy(request_payload)
    request["samples"] = [
        {"sample_id": f"sample-{i}", "text": "positive" if i % 2 else "negative"} for i in range(9)
    ]
    request["corpus_sha256"] = client.sha256_json(request["samples"])
    children = client.split_request(request)
    adapter = LayaAdapter(fake_agent, "/fixture", runtime=runtime_identity)
    records = [
        {
            "request": child,
            "request_sha256": client.sha256_json(child),
            "endpoint_image_digest": runtime_identity.configured_image_digest,
            "response": adapter.extract(ExtractionRequest.model_validate(child)).model_dump(
                mode="json"
            ),
        }
        for child in children
    ]
    cache = tmp_path / "window.json"
    client.atomic_write_json(
        cache, {"request_sha256": client.sha256_json(request), "subresponses": records[:1]}
    )
    assert client.load_cached_response(cache, request) is None
    client.atomic_write_json(
        cache, {"request_sha256": client.sha256_json(request), "subresponses": records}
    )
    merged = client.load_cached_response(cache, request)
    assert [sample["target_value"] for sample in merged["samples"]] == [
        -16.0,
        8.0,
        -16.0,
        8.0,
        -16.0,
        8.0,
        -16.0,
        8.0,
        -16.0,
    ]
    assert [part["response"]["batch_context_sha256"] for part in merged["execution_batches"]] == [
        client.sha256_json(child["samples"]) for child in children
    ]
    assert "batch_context_sha256" not in merged
    corrupted = copy.deepcopy(merged)
    corrupted["samples"][0]["target_value"] = 1.0
    with pytest.raises(ValueError):
        client.validate_response(request, corrupted)
