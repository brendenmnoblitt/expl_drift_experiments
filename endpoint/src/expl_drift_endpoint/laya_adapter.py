"""Score extraction through Laya's typed-decision SDK."""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any

from expl_drift_endpoint.contract import (
    CandidateScore,
    ExtractionRequest,
    ExtractionResponse,
    ModelRevision,
    SampleScores,
)

MODEL_ID = "convaiinnovations/laya"
MODEL_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
QUESTION_ID = "expl_drift_target"


class UnsupportedCapability(ValueError):
    """The loaded adapter cannot produce a requested kind of signal."""


class IncompatibleRevision(ValueError):
    """The request does not identify the checkpoint loaded by this process."""


class InvalidModelOutput(RuntimeError):
    """Laya returned a malformed or incompatible decision result."""


class LayaAdapter:
    """Use the pinned English base Laya checkpoint for fixed-option decisions."""

    def __init__(self, agent: Any, model_path: str, revision: str = MODEL_REVISION):
        self._agent = agent
        self.model_path = model_path
        self.revision = revision
        self.model_revision = ModelRevision(repository=MODEL_ID, revision=revision)
        self.tokenizer_revision = self.model_revision
        self.max_tokens = int(agent.cfg.get("max_len", 512))
        self._inference_lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> LayaAdapter:
        """Load the mounted model once; fail rather than silently serving on CPU."""
        model_path = os.environ.get("LAYA_MODEL_PATH", "/repository")
        revision = os.environ.get("LAYA_MODEL_REVISION", MODEL_REVISION)
        if revision != MODEL_REVISION:
            raise ValueError(f"Unsupported Laya revision {revision!r}; expected {MODEL_REVISION}")
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Laya model mount is unavailable: {model_path}")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the remote Laya adapter")

        from laya import Agent

        agent = Agent(model_id_or_path=model_path, device="cuda")
        return cls(agent=agent, model_path=model_path, revision=revision)

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """Return ordered candidate probabilities for each request sample."""
        if request.model != self.model_revision or request.tokenizer != self.tokenizer_revision:
            raise IncompatibleRevision(
                "request model/tokenizer revisions do not match the loaded Laya checkpoint"
            )
        if request.head is not None:
            raise UnsupportedCapability(
                "the base Laya checkpoint has no separately mounted decision head"
            )
        if request.attribution is not None and request.attribution.method != "integrated_gradients":
            raise UnsupportedCapability(
                "the implemented Laya attribution method is integrated_gradients"
            )
        if request.representations:
            raise UnsupportedCapability("Laya representation extraction has not been qualified")
        if request.max_tokens > self.max_tokens:
            raise UnsupportedCapability(
                f"requested max_tokens={request.max_tokens} exceeds "
                f"Laya's configured limit {self.max_tokens}"
            )

        criteria = {
            candidate.candidate_id: candidate.description for candidate in request.candidates
        }
        questions = {
            QUESTION_ID: {
                "type": "choice",
                "instructions": request.instructions,
                "criteria": criteria,
            }
        }
        with self._inference_lock:
            results = self._agent.predict_batch(
                [sample.text for sample in request.samples],
                questions,
                batch_size=len(request.samples),
                max_len=request.max_tokens,
            )
        if len(results) != len(request.samples):
            raise InvalidModelOutput("Laya returned a different number of results than requested")

        candidate_ids = [candidate.candidate_id for candidate in request.candidates]
        sample_scores: list[SampleScores] = []
        for sample, result in zip(request.samples, results, strict=True):
            try:
                answer = result["answers"][QUESTION_ID]
                predicted = answer["choice"]
                probabilities = answer["probabilities"]
            except (KeyError, TypeError) as error:
                raise InvalidModelOutput(
                    "Laya result is missing the typed choice output"
                ) from error

            if predicted not in candidate_ids or set(probabilities) != set(candidate_ids):
                raise InvalidModelOutput("Laya returned candidate IDs that differ from the request")
            scores = [
                CandidateScore(candidate_id=cid, probability=float(probabilities[cid]))
                for cid in candidate_ids
            ]
            values = [score.probability for score in scores]
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
                raise InvalidModelOutput("Laya returned a non-finite or out-of-range probability")
            if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=0.0001 * len(values)):
                raise InvalidModelOutput(
                    "Laya choice probabilities do not sum to one within SDK rounding"
                )
            attribution = None
            if request.attribution is not None:
                from expl_drift_endpoint.laya_attribution import explain_choice

                with self._inference_lock:
                    attribution = explain_choice(
                        self._agent,
                        sample.text,
                        request.instructions,
                        [
                            (candidate.candidate_id, candidate.description)
                            for candidate in request.candidates
                        ],
                        request.attribution.target_candidate_id,
                        max_tokens=request.max_tokens,
                        n_steps=request.attribution.n_steps,
                    )
            sample_scores.append(
                SampleScores(
                    sample_id=sample.sample_id,
                    predicted_candidate_id=predicted,
                    scores=scores,
                    attribution=attribution,
                )
            )

        return ExtractionResponse(
            schema_version="1",
            request_id=request.request_id,
            run_id=request.run_id,
            window_index=request.window_index,
            model=self.model_revision,
            tokenizer=self.tokenizer_revision,
            score_semantics="choice_probability",
            samples=sample_scores,
        )
