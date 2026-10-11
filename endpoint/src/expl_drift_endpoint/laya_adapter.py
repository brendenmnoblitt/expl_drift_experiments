"""Fixed-margin extraction through the pinned Laya SDK decision model."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any

from expl_drift_endpoint.contract import (
    CandidateScore,
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


def _runtime_identity(agent: Any, model_path: str) -> RuntimeIdentity:
    import torch

    root = Path(model_path)
    artifacts = [
        path
        for path in root.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
    ]
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
        checkpoint_artifact_sha256=_hash_files(root, artifacts),
        sdk_version=importlib.metadata.version("laya"),
        torch_version=torch.__version__,
        transformers_version=importlib.metadata.version("transformers"),
        python_version=platform.python_version(),
        device=str(agent.device),
        cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(agent.device) if agent.device.type == "cuda" else None,
        settings_sha256=hashlib.sha256(
            json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
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
        self.runtime = runtime if runtime is not None else _runtime_identity(agent, model_path)
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
            encode_batch,
            explain_margin,
            token_signal,
        )

        started = time.monotonic()
        candidate_ids = [candidate.candidate_id for candidate in request.candidates]
        target_index = candidate_ids.index(request.decision_target.target_candidate_id)
        reference_index = candidate_ids.index(request.decision_target.reference_candidate_id)
        outputs = []
        with self._inference_lock:
            items, batch = encode_batch(self._agent, request)
            with torch.no_grad():
                logits, hidden = _forward(self._agent, batch, capture=request.representations)
            if logits.shape != (len(request.samples), len(candidate_ids)):
                raise InvalidModelOutput("SDK raw-logit shape differs from ordered request")
            if not torch.isfinite(logits).all():
                raise InvalidModelOutput("SDK decision logits must be finite")
            for row, (sample, item) in enumerate(zip(request.samples, items, strict=True)):
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
                probabilities = torch.softmax(logits[row] / temperature, dim=-1).cpu().tolist()
                raw = logits[row].cpu().tolist()
                attribution = None
                if request.attribution is not None:
                    attribution = explain_margin(
                        self._agent,
                        batch,
                        tokens,
                        request.decision_target,
                        candidate_ids,
                        row,
                        logits,
                        n_steps=request.attribution.n_steps,
                    )
                representation = None
                if request.representations:
                    representation = RepresentationSignal(
                        pooling=request.pooling,
                        values=document_mean(hidden, row, tokens).detach().cpu().tolist(),
                    )
                outputs.append(
                    SampleScores(
                        sample_id=sample.sample_id,
                        predicted_candidate_id=candidate_ids[int(logits[row].argmax().item())],
                        scores=[
                            CandidateScore(
                                candidate_id=cid, logit=float(logit), probability=float(p)
                            )
                            for cid, logit, p in zip(candidate_ids, raw, probabilities, strict=True)
                        ],
                        probability_temperature=temperature,
                        target_value=float(
                            (logits[row, target_index] - logits[row, reference_index]).item()
                        ),
                        tokens=tokens,
                        attribution=attribution,
                        representation=representation,
                    )
                )
        return ExtractionResponse(
            schema_version="2",
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
