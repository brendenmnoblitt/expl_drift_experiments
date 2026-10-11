"""Fixed-margin IG and document pooling for pinned BERT/GPT-2 classifiers.

No model download or training is performed here. Local checkpoint loading hashes
all checkpoint files; qualification against actual task checkpoints is separate.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def checkpoint_fingerprint(directory: str | Path) -> str:
    """Hash file names and contents, including tokenizer/config/training metadata."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory is unavailable: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("checkpoint directory is empty")
    digest = hashlib.sha256()
    for path in files:
        name = path.relative_to(root).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class ReferenceAdapter:
    """Explain every non-reference candidate minus the same fixed reference.

    Float32/eval mode, right padding, document-only pad-embedding baseline,
    and last-layer document-mean pooling are explicit measurement semantics.
    GPT-2 pooling indices are held fixed along the IG path: deriving them from
    replaced baseline IDs, or from inputs_embeds alone, changes the function.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        candidate_ids: Sequence[str],
        reference_candidate_id: str,
        checkpoint_sha256: str,
        tokenizer_sha256: str,
        max_tokens: int = 128,
        n_steps: int = 64,
        completeness_atol: float = 0.01,
        completeness_rtol: float = 0.01,
        forward_atol: float = 1e-5,
        forward_rtol: float = 1e-5,
    ) -> None:
        import torch
        from expl_drift_endpoint import laya_attribution

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.candidate_ids = tuple(candidate_ids)
        if len(self.candidate_ids) < 2 or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate IDs must be ordered and unique")
        if reference_candidate_id not in self.candidate_ids:
            raise ValueError("reference candidate must belong to the task")
        if model.config.num_labels != len(self.candidate_ids):
            raise ValueError("checkpoint classifier head does not match the task schema")
        self.reference_index = self.candidate_ids.index(reference_candidate_id)
        self.model_type = model.config.model_type
        if self.model_type not in ("bert", "gpt2"):
            raise ValueError("only BERT and GPT-2 reference classifiers are supported")
        if self.model_type == "gpt2":
            model.config.use_cache = False
        if tokenizer.pad_token_id is None or tokenizer.padding_side != "right":
            raise ValueError("qualification requires explicit pad token and right padding")
        if model.config.pad_token_id != tokenizer.pad_token_id:
            raise ValueError("checkpoint and tokenizer pad identities differ")
        if max_tokens < 2 or n_steps < 2 or n_steps > 64:
            raise ValueError("max_tokens >= 2 and 2 <= n_steps <= 64 are required")
        for digest in (checkpoint_sha256, tokenizer_sha256):
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("checkpoint and tokenizer require SHA-256 identities")
        tolerances = (completeness_atol, completeness_rtol, forward_atol, forward_rtol)
        if any(not math.isfinite(value) or value < 0 for value in tolerances):
            raise ValueError("numerical tolerances must be finite and nonnegative")
        if any(
            param.dtype != torch.float32
            for param in model.parameters()
            if param.is_floating_point()
        ):
            raise ValueError("reference extraction is pinned to float32 weights")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.max_tokens, self.n_steps = max_tokens, n_steps
        self.completeness_atol, self.completeness_rtol = completeness_atol, completeness_rtol
        self.forward_atol, self.forward_rtol = forward_atol, forward_rtol
        self.device = next(model.parameters()).device
        self.runtime = {
            "model_type": self.model_type,
            "checkpoint_sha256": checkpoint_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "observed_checkpoint_label_mapping": {
                str(index): label for index, label in model.config.id2label.items()
            },
            "precision": "float32",
            "pooling": "document_mean_last_hidden",
            "baseline": "pad_document_tokens",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "integration_implementation_sha256": hashlib.sha256(
                Path(laya_attribution.__file__).read_bytes()
            ).hexdigest(),
            "tf32": False,
            "target_semantics": "fixed_candidate_margin",
            "candidate_ids": list(self.candidate_ids),
            "reference_candidate_id": reference_candidate_id,
            "max_tokens": max_tokens,
            "n_steps": n_steps,
            "padding_side": "right",
            "device": str(self.device),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "completeness_atol": completeness_atol,
            "completeness_rtol": completeness_rtol,
            "forward_atol": forward_atol,
            "forward_rtol": forward_rtol,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        tokenizer_path: str | Path | None = None,
        device: str = "cpu",
        **settings: Any,
    ) -> ReferenceAdapter:
        """Load only local, checksummed task artifacts; never substitute base weights."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        checkpoint = Path(checkpoint)
        tokenizer_path = Path(tokenizer_path) if tokenizer_path is not None else checkpoint
        checkpoint_id = checkpoint_fingerprint(checkpoint)
        tokenizer_id = checkpoint_fingerprint(tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            local_files_only=True,
            dtype="float32",
        ).to(device)
        return cls(
            model,
            tokenizer,
            checkpoint_sha256=checkpoint_id,
            tokenizer_sha256=tokenizer_id,
            **settings,
        )

    def _forward(
        self, embeddings: Any, encoded: dict[str, Any], pool_indices: Any
    ) -> tuple[Any, Any]:
        """Reuse the actual classifier head, with fixed GPT-2 sequence indices."""
        import torch

        with torch.autocast(device_type=self.device.type, enabled=False):
            if self.model_type == "bert":
                outputs = self.model(
                    inputs_embeds=embeddings,
                    **encoded,
                    output_hidden_states=True,
                )
                return outputs.logits, outputs.hidden_states[-1]
            outputs = self.model.transformer(
                inputs_embeds=embeddings,
                **encoded,
                use_cache=False,
            )
            hidden = outputs.last_hidden_state
            all_logits = self.model.score(hidden)
            return all_logits[torch.arange(len(hidden), device=self.device), pool_indices], hidden

    def extract(
        self, texts: Sequence[str], sample_ids: Sequence[str], *, batch_size: int = 1
    ) -> dict[str, Any]:
        """Save full ordered decision signals and all fixed margins; labels stay local."""
        import torch
        from expl_drift_endpoint.laya_attribution import integrate_gradients

        if len(texts) != len(sample_ids) or not texts or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("texts and unique ordered sample IDs must be nonempty and aligned")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        samples = []
        started = time.monotonic()
        embedding_layer = self.model.get_input_embeddings()
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            batch_started = time.monotonic()
            tokens = self.tokenizer(
                chunk,
                padding="max_length",
                truncation=True,
                max_length=self.max_tokens,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special = tokens.pop("special_tokens_mask").to(self.device)
            ids = tokens.pop("input_ids").to(self.device)
            encoded = {key: value.to(self.device) for key, value in tokens.items()}
            document = encoded["attention_mask"].bool() & ~special.bool()
            positions = torch.arange(ids.shape[1], device=self.device)
            if self.model_type == "gpt2":
                nonpad = ids.ne(self.model.config.pad_token_id)
                if not nonpad.any(dim=1).all():
                    raise ValueError("GPT-2 input has no classifier pooling token")
                pool_indices = (nonpad * positions).max(dim=1).values
            else:
                pool_indices = encoded["attention_mask"].sum(dim=1) - 1
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
                embeddings = embedding_layer(ids).detach()
                batch_logits, hidden = self._forward(embeddings, encoded, pool_indices)
                id_logits = self.model(input_ids=ids, **encoded).logits
                if not torch.allclose(
                    batch_logits, id_logits, atol=self.forward_atol, rtol=self.forward_rtol
                ):
                    raise RuntimeError(
                        "input-ID scoring and attributed embedding function disagree"
                    )
                probabilities = torch.softmax(batch_logits, dim=-1)
                counts = document.sum(dim=1)
                pooled = (hidden * document.unsqueeze(-1)).sum(dim=1) / counts.clamp_min(
                    1
                ).unsqueeze(-1)
                baseline_ids = torch.where(document, self.tokenizer.pad_token_id, ids)
                baselines = embedding_layer(baseline_ids).detach()
            if not torch.isfinite(batch_logits).all() or not torch.isfinite(pooled).all():
                raise RuntimeError("nonfinite classifier or pooled representation")
            score_latency = time.monotonic() - batch_started
            for offset, text in enumerate(chunk):
                sample_started = time.monotonic()
                row = slice(offset, offset + 1)
                row_encoded = {key: value[row] for key, value in encoded.items()}
                row_indices = pool_indices[row]
                with torch.no_grad():
                    input_logits, _ = self._forward(embeddings[row], row_encoded, row_indices)
                    baseline_logits, _ = self._forward(baselines[row], row_encoded, row_indices)
                if (
                    not torch.isfinite(input_logits).all()
                    or not torch.isfinite(baseline_logits).all()
                ):
                    raise RuntimeError("nonfinite reference forward quantities")
                targets = []
                for target_index, candidate_id in enumerate(self.candidate_ids):
                    if target_index == self.reference_index:
                        continue
                    input_value = float(
                        (
                            input_logits[0, target_index] - input_logits[0, self.reference_index]
                        ).item()
                    )
                    baseline_value = float(
                        (
                            baseline_logits[0, target_index]
                            - baseline_logits[0, self.reference_index]
                        ).item()
                    )
                    delta = input_value - baseline_value
                    tolerance = self.completeness_atol + self.completeness_rtol * abs(delta)

                    def gradient_at(point: Any) -> Any:
                        point = point.detach().requires_grad_(True)
                        logits, _ = self._forward(point, row_encoded, row_indices)
                        margin = logits[0, target_index] - logits[0, self.reference_index]
                        return torch.autograd.grad(margin, point)[0].detach()

                    attrs, evaluations = integrate_gradients(
                        embeddings[row],
                        baselines[row],
                        gradient_at,
                        self.n_steps,
                        target_delta=delta,
                        completeness_tolerance=tolerance,
                    )
                    values = attrs.sum(dim=-1)[0]
                    residual = delta - float(values.sum().item())
                    if not torch.isfinite(values).all() or abs(residual) > tolerance:
                        raise RuntimeError("reference IG failed completeness")
                    targets.append(
                        {
                            "target_candidate_id": candidate_id,
                            "reference_candidate_id": self.candidate_ids[self.reference_index],
                            "target_semantics": "fixed_candidate_margin",
                            "values": values.cpu().tolist(),
                            "input_target_value": input_value,
                            "baseline_target_value": baseline_value,
                            "completeness_delta": residual,
                            "gradient_evaluations": evaluations,
                        }
                    )
                full = self.tokenizer(text, truncation=False, return_special_tokens_mask=True)
                total = sum(value == 0 for value in full["special_tokens_mask"])
                retained = int(counts[offset].item())
                samples.append(
                    {
                        "sample_id": sample_ids[start + offset],
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "candidate_ids": list(self.candidate_ids),
                        "logits": batch_logits[offset].cpu().tolist(),
                        "probabilities": probabilities[offset].cpu().tolist(),
                        "attribution_input_logits": input_logits[0].cpu().tolist(),
                        "baseline_logits": baseline_logits[0].cpu().tolist(),
                        "singleton_batch_max_logit_discrepancy": float(
                            (input_logits[0] - batch_logits[offset]).abs().max().item()
                        ),
                        "token_ids": ids[offset].cpu().tolist(),
                        "tokens": self.tokenizer.convert_ids_to_tokens(ids[offset].cpu().tolist()),
                        "segments": [
                            "document" if doc else "special" if active else "padding"
                            for doc, active in zip(
                                document[offset].cpu().tolist(),
                                encoded["attention_mask"][offset].cpu().tolist(),
                                strict=True,
                            )
                        ],
                        "document_mask": document[offset].cpu().tolist(),
                        "retained_document_tokens": retained,
                        "total_document_tokens": total,
                        "dropped_document_tokens": total - retained,
                        "scoring_batch_latency_seconds": score_latency,
                        "attribution_latency_seconds": time.monotonic() - sample_started,
                        "representation": pooled[offset].cpu().tolist(),
                        "attributions": targets,
                    }
                )
        return {
            "schema_version": "research-reference-1",
            "runtime": self.runtime,
            "batch_context": {"batch_size": batch_size, "ordered_sample_ids": list(sample_ids)},
            "latency_seconds": time.monotonic() - started,
            "samples": samples,
        }
