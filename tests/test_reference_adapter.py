"""Real tiny HF networks exercise padded classifier functions, not checkpoints."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    GPT2Config,
    GPT2ForSequenceClassification,
    PreTrainedTokenizerFast,
)

from experiments.research_extension.reference_adapter import ReferenceAdapter


def tiny_reference(model_type):
    torch.manual_seed(17)
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "hello": 4, "world": 5, "!": 6}
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    if model_type == "bert":
        raw.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
        )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    if model_type == "bert":
        model = BertForSequenceClassification(
            BertConfig(
                vocab_size=7,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_labels=3,
                pad_token_id=0,
                max_position_embeddings=16,
                hidden_dropout_prob=0,
                attention_probs_dropout_prob=0,
            )
        )
    else:
        model = GPT2ForSequenceClassification(
            GPT2Config(
                vocab_size=7,
                n_embd=8,
                n_layer=1,
                n_head=2,
                num_labels=3,
                pad_token_id=0,
                bos_token_id=2,
                eos_token_id=3,
                n_positions=16,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
            )
        )
    adapter = ReferenceAdapter(
        model,
        tokenizer,
        candidate_ids=["reference", "a", "b"],
        reference_candidate_id="reference",
        checkpoint_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        max_tokens=8,
        n_steps=16,
    )
    return adapter


@pytest.mark.parametrize("model_type", ["bert", "gpt2"])
def test_padded_reference_margins_explain_actual_classifier_logits(model_type):
    adapter = tiny_reference(model_type)
    texts = ["hello", "hello world ! hello world ! hello world !"]
    bank = adapter.extract(texts, ["short", "truncated"], batch_size=2)
    encoded = adapter.tokenizer(
        texts, padding="max_length", truncation=True, max_length=8, return_tensors="pt"
    )
    with torch.no_grad():
        expected = adapter.model(**encoded).logits.numpy()
    for index, sample in enumerate(bank["samples"]):
        np.testing.assert_allclose(sample["logits"], expected[index], atol=1e-5)
        np.testing.assert_allclose(
            sample["probabilities"], torch.softmax(torch.tensor(expected[index]), -1), atol=1e-6
        )
        for target in sample["attributions"]:
            target_index = adapter.candidate_ids.index(target["target_candidate_id"])
            assert target["input_target_value"] == pytest.approx(
                expected[index, target_index] - expected[index, 0], abs=1e-5
            )
            residual = (
                target["input_target_value"]
                - target["baseline_target_value"]
                - sum(target["values"])
            )
            assert residual == pytest.approx(target["completeness_delta"], abs=1e-7)
            assert abs(residual) <= 0.01 + 0.01 * abs(
                target["input_target_value"] - target["baseline_target_value"]
            )
            assert all(
                value == 0
                for value, keep in zip(target["values"], sample["document_mask"], strict=True)
                if not keep
            )
    assert bank["samples"][1]["dropped_document_tokens"] > 0


def test_gpt2_margin_target_does_not_follow_changed_prediction():
    adapter = tiny_reference("gpt2")
    original = adapter.extract(["hello world"], ["sample"])["samples"][0]
    with torch.no_grad():
        adapter.model.score.weight.neg_()
    reversed_sample = adapter.extract(["hello world"], ["sample"])["samples"][0]
    assert np.argmax(original["logits"]) != np.argmax(reversed_sample["logits"])
    for original_target, reversed_target in zip(
        original["attributions"], reversed_sample["attributions"], strict=True
    ):
        assert reversed_target["reference_candidate_id"] == "reference"
        assert reversed_target["input_target_value"] == pytest.approx(
            -original_target["input_target_value"], abs=1e-6
        )
        np.testing.assert_allclose(
            reversed_target["values"], -np.array(original_target["values"]), atol=1e-6
        )
