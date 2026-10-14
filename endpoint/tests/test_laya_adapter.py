"""CPU autograd fixtures qualify declared functions, not deployed checkpoints."""

import copy

import numpy as np
import pytest

from expl_drift_endpoint.contract import ExtractionRequest
from expl_drift_endpoint.laya_adapter import (
    IncompatibleRevision,
    LayaAdapter,
    _checkpoint_file_manifest,
    _hash_model_state,
    _hash_tokenizer,
)


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


def test_sample_outputs_are_invariant_to_request_batch_grouping(
    request_payload, runtime_identity, fake_agent
):
    adapter = LayaAdapter(fake_agent, "/fixture", runtime=runtime_identity)
    grouped = adapter.extract(ExtractionRequest.model_validate(request_payload))

    for index, grouped_sample in enumerate(grouped.samples):
        single_payload = copy.deepcopy(request_payload)
        single_payload["request_id"] = f"request-single-{index}"
        single_payload["samples"] = [request_payload["samples"][index]]
        single = adapter.extract(ExtractionRequest.model_validate(single_payload))

        assert grouped_sample.model_dump(mode="json") == single.samples[0].model_dump(mode="json")


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


def test_loaded_model_state_fingerprint_is_content_stable_and_sensitive():
    import torch
    from torch import nn

    first = nn.Linear(2, 3)
    second = nn.Linear(2, 3)
    second.load_state_dict(first.state_dict())
    assert _hash_model_state(first) == _hash_model_state(second)

    with torch.no_grad():
        second.weight[0, 0] += 1
    assert _hash_model_state(first) != _hash_model_state(second)


def test_checkpoint_manifest_identifies_the_changed_artifact(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    config = tokenizer_dir / "tokenizer_config.json"
    config.write_text('{"tokenizer_class":"A"}', encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    before_digest, before_files = _checkpoint_file_manifest(tmp_path)
    config.write_text('{"tokenizer_class":"B"}', encoding="utf-8")
    after_digest, after_files = _checkpoint_file_manifest(tmp_path)

    assert before_digest != after_digest
    changed_paths = [
        before.path
        for before, after in zip(before_files, after_files, strict=True)
        if before.sha256 != after.sha256
    ]
    assert changed_paths == ["tokenizer/tokenizer_config.json"]


def test_tokenizer_fingerprint_includes_backend_vocabulary():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    def make_tokenizer():
        backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")

    first = make_tokenizer()
    second = make_tokenizer()
    assert _hash_tokenizer(first) == _hash_tokenizer(second)
    second.add_tokens(["new_token"])
    assert _hash_tokenizer(first) != _hash_tokenizer(second)
