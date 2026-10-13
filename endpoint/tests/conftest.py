"""Synthetic CPU architecture fixtures; no checkpoint qualification is implied."""

from types import SimpleNamespace

import pytest

from expl_drift_endpoint.contract import RuntimeIdentity


@pytest.fixture
def runtime_identity():
    return RuntimeIdentity(
        implementation_sha256="1" * 64,
        checkpoint_artifact_sha256="2" * 64,
        sdk_version="0.3.29",
        torch_version="fixture",
        transformers_version="fixture",
        python_version="fixture",
        device="cpu",
        cuda_version=None,
        gpu_name=None,
        settings_sha256="3" * 64,
        configured_image_digest="sha256:" + "4" * 64,
    )


@pytest.fixture
def request_payload(runtime_identity):
    revision = {
        "repository": "convaiinnovations/laya",
        "revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
    }
    return {
        "schema_version": "2",
        "request_id": "request-1",
        "run_id": "run-1",
        "window_index": 0,
        "corpus_sha256": "b" * 64,
        "model": revision,
        "tokenizer": revision,
        "instructions": "Choose the matching category.",
        "candidates": [
            {"candidate_id": "alpha", "description": "First category"},
            {"candidate_id": "beta", "description": "Second category"},
        ],
        "samples": [
            {"sample_id": "sample-1", "text": "positive"},
            {"sample_id": "sample-2", "text": "negative"},
        ],
        "max_tokens": 8,
        "decision_target": {
            "target_semantics": "fixed_candidate_margin",
            "target_candidate_id": "beta",
            "reference_candidate_id": "alpha",
        },
        "precision": "float32_no_autocast",
        "tokenization": "laya_sdk_0.3.29_encode_state",
        "pooling": "document_mean_last_hidden_state",
        "runtime": runtime_identity.model_dump(),
        "attribution": {"method": "integrated_gradients", "n_steps": 16},
        "representations": True,
    }


@pytest.fixture
def fake_agent(monkeypatch):
    import sys

    import torch
    from torch import nn

    class Embeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.words = nn.Embedding(7, 2)
            with torch.no_grad():
                self.words.weight.copy_(
                    torch.tensor(
                        [
                            [0.0, 0.0],
                            [100.0, 200.0],
                            [50.0, 100.0],
                            [60.0, 120.0],
                            [2.0, 1.0],
                            [-1.0, -0.5],
                            [300.0, 600.0],
                        ]
                    )
                )

        def forward(self, input_ids=None, inputs_embeds=None):
            return self.words(input_ids) if inputs_embeds is None else inputs_embeds

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = Embeddings()

        def get_input_embeddings(self):
            return self.embeddings.words

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embeddings(input_ids=input_ids) * 2)

    class Decision(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            hidden = self.encoder(input_ids, attention_mask).last_hidden_state
            # Synthetic backend sensitivity makes request grouping observable.
            value = hidden[:, 3:5, 0].sum(dim=1) * (1.0 + 0.001 * (len(input_ids) - 1))
            return torch.stack([-value, value * 3], dim=1), torch.zeros((len(value), 2))

    class Tokenizer:
        pad_token_id = 0
        all_special_ids = [0, 1, 6]

        def convert_ids_to_tokens(self, ids):
            return [f"token-{value}" for value in ids]

    class Agent:
        cfg = {"max_len": 8, "head_max_len": 192}
        device = torch.device("cpu")
        tok = Tokenizer()
        temperature = [2.0, 2.0, 2.0]
        temperature_by_options = {}
        parallel_options = False
        model = Decision()

        def _to_internal(self, question):
            assert list(question["criteria"]) == ["alpha", "beta"]
            return question

        def _encode_state(self, state, ids, internal, *, max_len, head_max_len):
            assert max_len == 8
            document = [4, 5] if state == "positive" else [5, 5]
            return [
                {
                    "ids": [1, 2, 3, *document, 6],
                    "markers": [1, 2],
                    "qtype": 0,
                    "state_stats": {
                        "state_tokens_used": 2,
                        "state_tokens": 3,
                        "state_tokens_dropped": 1,
                    },
                }
            ]

    def collate_items(groups, pad_id):
        items = [item for group in groups for item in group]
        return {
            "input_ids": torch.tensor([item["ids"] for item in items]),
            "attention_mask": torch.ones((len(items), 6), dtype=torch.long),
            "marker_pos": torch.tensor([item["markers"] for item in items]),
            "marker_mask": torch.ones((len(items), 2), dtype=torch.bool),
            "qtype": torch.zeros(len(items), dtype=torch.long),
        }

    # Substitute only SDK sequence mechanics; execute real torch autograd/model hooks.
    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "laya.common",
        SimpleNamespace(
            collate_items=collate_items,
            temp_bucket=lambda qt, count: f"{qt}:{count}",
        ),
    )
    return Agent()


@pytest.fixture
def response_payload(request_payload, runtime_identity, fake_agent):
    from expl_drift_endpoint.contract import ExtractionRequest
    from expl_drift_endpoint.laya_adapter import LayaAdapter

    adapter = LayaAdapter(fake_agent, "/fixture", runtime=runtime_identity)
    return adapter.extract(ExtractionRequest.model_validate(request_payload)).model_dump(
        mode="json"
    )
