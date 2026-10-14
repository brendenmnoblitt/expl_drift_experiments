"""Fixed-margin extraction through the pinned Laya SDK decision model."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from expl_drift_endpoint.contract import (
    CandidateScore,
    CheckpointFileIdentity,
    CheckpointMountManifest,
    ExtractionRequest,
    ExtractionResponse,
    ModelRevision,
    RepresentationSignal,
    RuntimeIdentity,
    SampleScores,
    request_fingerprint,
)

MODEL_ID = "convaiinnovations/laya"
MODEL_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
QUESTION_ID = "expl_drift_target"


class UnsupportedCapability(ValueError):
    """The loaded adapter cannot produce a requested kind of signal."""


class IncompatibleRevision(ValueError):
    """The request does not identify the runtime loaded by this process."""


class InvalidModelOutput(RuntimeError):
    """Laya returned a malformed or incompatible decision result."""


def _hash_files(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        name = path.relative_to(root).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    if not paths:
        raise ValueError(f"No runtime artifact files found in {root}")
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_file_manifest(root: Path) -> tuple[str, list[CheckpointFileIdentity]]:
    """Return the legacy tree digest and per-file hashes for startup diagnostics."""
    digest = hashlib.sha256()
    files: list[CheckpointFileIdentity] = []
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
    ]
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        name = relative.encode()
        size = path.stat().st_size
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(size.to_bytes(8, "big"))
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                file_digest.update(chunk)
        files.append(
            CheckpointFileIdentity(path=relative, size_bytes=size, sha256=file_digest.hexdigest())
        )
    if not files:
        raise ValueError(f"No runtime artifact files found in {root}")
    return digest.hexdigest(), files


def _hash_model_state(model: Any) -> str:
    """Hash loaded tensor values and metadata."""
    import torch

    digest = hashlib.sha256()
    state = model.state_dict()
    if not state:
        raise ValueError("Loaded model has no state tensors")
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Model state entry {name!r} is not a tensor")
        encoded_name = name.encode()
        metadata = json.dumps(
            [str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        value = tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(value))
    return digest.hexdigest()


def _tokenizer_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _tokenizer_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_tokenizer_value(item) for item in value]
    if hasattr(value, "content"):
        return {
            name: getattr(value, name)
            for name in ("content", "single_word", "lstrip", "rstrip", "normalized", "special")
            if hasattr(value, name)
        }
    raise TypeError(f"Unsupported tokenizer identity value: {type(value).__qualname__}")


def _hash_tokenizer(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not callable(getattr(backend, "to_str", None)):
        raise TypeError("Laya tokenizer must expose its serialized backend tokenizer")
    return _hash_json(
        {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "backend": backend.to_str(),
            "special_tokens_map": _tokenizer_value(tokenizer.special_tokens_map),
            "model_input_names": list(tokenizer.model_input_names),
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "model_max_length": tokenizer.model_max_length,
        }
    )


def _runtime_identity(agent: Any) -> RuntimeIdentity:
    import torch

    source_root = Path(__file__).parent
    settings = {
        "cfg": agent.cfg,
        "temperature": agent.temperature,
        "temperature_by_options": agent.temperature_by_options,
        "parallel_options": agent.parallel_options,
        "precision": "float32_no_autocast",
        "tf32": False,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    return RuntimeIdentity(
        implementation_sha256=_hash_files(source_root, list(source_root.glob("*.py"))),
        model_state_sha256=_hash_model_state(agent.model),
        model_config_sha256=_hash_json(
            {
                "laya_config": agent.cfg,
                "encoder_config": agent.model.encoder.config.to_dict(),
            }
        ),
        tokenizer_sha256=_hash_tokenizer(agent.tok),
        sdk_version=importlib.metadata.version("laya"),
        torch_version=torch.__version__,
        transformers_version=importlib.metadata.version("transformers"),
        python_version=platform.python_version(),
        device=str(agent.device),
        cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(agent.device) if agent.device.type == "cuda" else None,
        settings_sha256=_hash_json(settings),
        configured_image_digest=os.environ["LAYA_ENDPOINT_IMAGE_DIGEST"],
    )


class LayaAdapter:
    """Expose implemented signals, not a claim of scientific qualification."""

    capabilities = [
        "decision_scores",
        "fixed_candidate_margin",
        "integrated_gradients",
        "document_mean_representation",
        "exact_runtime_fingerprint",
    ]

    def __init__(
        self,
        agent: Any,
        model_path: str,
        revision: str = MODEL_REVISION,
        *,
        runtime: RuntimeIdentity | None = None,
        checkpoint_mount: CheckpointMountManifest | None = None,
    ):
        import torch

        self._agent = agent
        self.model_path = model_path
        self.revision = revision
        self.model_revision = ModelRevision(repository=MODEL_ID, revision=revision)
        self.tokenizer_revision = self.model_revision
        self.max_tokens = int(agent.cfg["max_len"])
        agent.model.float().eval()
        if any(parameter.dtype != torch.float32 for parameter in agent.model.parameters()):
            raise ValueError("Laya extraction requires float32 model parameters")
        # TF32 would contradict the declared full-precision arithmetic path.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.runtime = runtime if runtime is not None else _runtime_identity(agent)
        if checkpoint_mount is None:
            artifact_sha, files = _checkpoint_file_manifest(Path(model_path))
            checkpoint_mount = CheckpointMountManifest(
                checkpoint_artifact_sha256=artifact_sha,
                checkpoint_files=files,
            )
        self.checkpoint_mount = checkpoint_mount
        self._inference_lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> LayaAdapter:
        model_path = os.environ.get("LAYA_MODEL_PATH", "/repository")
        revision = os.environ.get("LAYA_MODEL_REVISION", MODEL_REVISION)
        if revision != MODEL_REVISION:
            raise ValueError(f"Unsupported Laya revision {revision!r}; expected {MODEL_REVISION}")
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Laya model mount is unavailable: {model_path}")
        if importlib.metadata.version("laya") != "0.3.29":
            raise ValueError("This adapter requires the source-qualified laya==0.3.29 SDK")
        import torch
        from laya import Agent

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the remote Laya adapter")
        agent = Agent(model_id_or_path=model_path, device="cuda")
        return cls(agent=agent, model_path=model_path, revision=revision)

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        if request.model != self.model_revision or request.tokenizer != self.tokenizer_revision:
            raise IncompatibleRevision("request model/tokenizer revisions differ from loaded Laya")
        if request.runtime != self.runtime:
            raise IncompatibleRevision("request runtime identity differs from loaded Laya runtime")
        if request.head is not None:
            raise UnsupportedCapability("base Laya has no separately mounted decision head")
        if request.max_tokens > self.max_tokens:
            raise UnsupportedCapability("requested token budget exceeds Laya's configured limit")

        import torch
        from laya.common import temp_bucket

        from expl_drift_endpoint.laya_attribution import (
            _forward,
            document_mean,
            encode_samples,
            explain_margin,
            token_signal,
        )

        started = time.monotonic()
        candidate_ids = [candidate.candidate_id for candidate in request.candidates]
        target_index = candidate_ids.index(request.decision_target.target_candidate_id)
        reference_index = candidate_ids.index(request.decision_target.reference_candidate_id)
        outputs = []
        with self._inference_lock:
            items, sample_batches = encode_samples(self._agent, request)
            for sample, item, batch in zip(request.samples, items, sample_batches, strict=True):
                with torch.no_grad():
                    logits, hidden = _forward(self._agent, batch, capture=request.representations)
                if logits.shape != (1, len(candidate_ids)):
                    raise InvalidModelOutput("SDK raw-logit shape differs from ordered request")
                if not torch.isfinite(logits).all():
                    raise InvalidModelOutput("SDK decision logits must be finite")
                tokens = token_signal(self._agent, item, candidate_ids)
                qt = item["qtype"]
                temperature = float(
                    self._agent.temperature_by_options.get(
                        temp_bucket(qt, len(candidate_ids)), self._agent.temperature[qt]
                    )
                )
                if not 0.0 < temperature < float("inf"):
                    raise InvalidModelOutput(
                        "SDK probability temperature must be finite and positive"
                    )
                probabilities = torch.softmax(logits[0] / temperature, dim=-1).cpu().tolist()
                raw = logits[0].cpu().tolist()
                attribution = None
                if request.attribution is not None:
                    attribution = explain_margin(
                        self._agent,
                        batch,
                        tokens,
                        request.decision_target,
                        candidate_ids,
                        logits,
                        n_steps=request.attribution.n_steps,
                    )
                representation = None
                if request.representations:
                    representation = RepresentationSignal(
                        pooling=request.pooling,
                        values=document_mean(hidden, 0, tokens).detach().cpu().tolist(),
                    )
                outputs.append(
                    SampleScores(
                        sample_id=sample.sample_id,
                        predicted_candidate_id=candidate_ids[int(logits[0].argmax().item())],
                        scores=[
                            CandidateScore(
                                candidate_id=cid, logit=float(logit), probability=float(p)
                            )
                            for cid, logit, p in zip(candidate_ids, raw, probabilities, strict=True)
                        ],
                        probability_temperature=temperature,
                        target_value=float(
                            (logits[0, target_index] - logits[0, reference_index]).item()
                        ),
                        tokens=tokens,
                        attribution=attribution,
                        representation=representation,
                    )
                )
        return ExtractionResponse(
            schema_version="5",
            request_id=request.request_id,
            request_sha256=request_fingerprint(request),
            run_id=request.run_id,
            window_index=request.window_index,
            model=self.model_revision,
            tokenizer=self.tokenizer_revision,
            runtime=self.runtime,
            decision_target=request.decision_target,
            precision=request.precision,
            tokenization=request.tokenization,
            pooling=request.pooling,
            score_semantics="temperature_scaled_choice_probability",
            batch_context_sha256=hashlib.sha256(
                json.dumps(
                    [sample.model_dump(mode="json") for sample in request.samples],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            latency_seconds=time.monotonic() - started,
            samples=outputs,
        )
