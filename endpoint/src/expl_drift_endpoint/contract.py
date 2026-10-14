"""Version 2 extraction: fixed decisions and exactly identified numerical paths."""

import hashlib
import json
import math
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MAX_IG_COMPUTE_STEPS = 16384
PRECISION = "float32_no_autocast"
TOKENIZATION = "laya_sdk_0.3.29_encode_state"
POOLING = "document_mean_last_hidden_state"

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
Commit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Fingerprint = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ImageDigest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


def request_fingerprint(request: "ExtractionRequest") -> str:
    """Fingerprint the full request batch; model execution is singleton per sample."""
    payload = json.dumps(
        request.model_dump(mode="json", exclude_unset=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ModelRevision(ContractModel):
    repository: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")]
    revision: Commit


class CheckpointFileIdentity(ContractModel):
    path: str
    size_bytes: int = Field(ge=0)
    sha256: Fingerprint


class RuntimeIdentity(ContractModel):
    """Observed software, checkpoint, tokenizer, model config, and loaded state."""

    implementation_sha256: Fingerprint
    checkpoint_artifact_sha256: Fingerprint
    checkpoint_files: list[CheckpointFileIdentity]
    model_state_sha256: Fingerprint
    model_config_sha256: Fingerprint
    tokenizer_sha256: Fingerprint
    sdk_version: str
    torch_version: str
    transformers_version: str
    python_version: str
    device: str
    cuda_version: str | None
    gpu_name: str | None
    settings_sha256: Fingerprint
    configured_image_digest: ImageDigest
    image_identity_semantics: Literal["configured_not_attested"] = "configured_not_attested"


class Sample(ContractModel):
    sample_id: Identifier
    text: str = Field(min_length=1, max_length=32768)


class Candidate(ContractModel):
    candidate_id: Identifier
    description: str = Field(min_length=1, max_length=2048)


class DecisionTarget(ContractModel):
    target_semantics: Literal["fixed_candidate_margin"]
    target_candidate_id: Identifier
    reference_candidate_id: Identifier

    @model_validator(mode="after")
    def distinct_candidates(self) -> Self:
        if self.target_candidate_id == self.reference_candidate_id:
            raise ValueError("margin target and reference must be distinct")
        return self


class AttributionPlan(ContractModel):
    method: Literal["integrated_gradients"]
    n_steps: int = Field(default=64, ge=2, le=64)


class ExtractionRequest(ContractModel):
    schema_version: Literal["4"]
    request_id: Identifier
    run_id: Identifier
    window_index: int = Field(ge=0)
    corpus_sha256: Fingerprint
    model: ModelRevision
    tokenizer: ModelRevision
    head: ModelRevision | None = None
    instructions: str = Field(min_length=1, max_length=4096)
    candidates: list[Candidate] = Field(min_length=2, max_length=64)
    samples: list[Sample] = Field(min_length=1, max_length=32)
    max_tokens: int = Field(ge=1, le=8192)
    decision_target: DecisionTarget
    precision: Literal["float32_no_autocast"]
    tokenization: Literal["laya_sdk_0.3.29_encode_state"]
    pooling: Literal["document_mean_last_hidden_state"]
    runtime: RuntimeIdentity
    attribution: AttributionPlan | None = None
    representations: bool = False

    @model_validator(mode="after")
    def validate_identities(self) -> Self:
        sample_ids = [sample.sample_id for sample in self.samples]
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample IDs must be unique")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique")
        if any(
            candidate not in candidate_ids
            for candidate in (
                self.decision_target.target_candidate_id,
                self.decision_target.reference_candidate_id,
            )
        ):
            raise ValueError("decision target and reference must identify supplied candidates")
        return self


class CandidateScore(ContractModel):
    candidate_id: Identifier
    logit: float = Field(allow_inf_nan=False)
    probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class TokenSignal(ContractModel):
    """The exact SDK sequence plus declared fixed-width storage padding."""

    token_width: int = Field(ge=1, le=8192)
    token_ids: list[int]
    tokens: list[str]
    segments: list[str]
    state_token_start: int = Field(ge=0)
    state_token_count: int = Field(ge=0)
    state_tokens_total: int = Field(ge=0)
    state_tokens_dropped: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        if {len(self.token_ids), len(self.tokens), len(self.segments)} != {self.token_width}:
            raise ValueError("token IDs, strings, and segments must match token_width")
        start, count = self.state_token_start, self.state_token_count
        if start + count > self.token_width or self.state_tokens_total < count:
            raise ValueError("document token span exceeds token_width or total")
        if self.state_tokens_dropped != self.state_tokens_total - count:
            raise ValueError("dropped token count must match total minus retained state tokens")
        if [i for i, segment in enumerate(self.segments) if segment == "document"] != list(
            range(start, start + count)
        ):
            raise ValueError("document segments must equal the retained document span")
        return self


class AttributionSignal(ContractModel):
    method: Literal["integrated_gradients"]
    target_semantics: Literal["fixed_candidate_margin"]
    target_candidate_id: Identifier
    reference_candidate_id: Identifier
    baseline: Literal["pad_document_tokens"]
    n_steps_requested: int = Field(ge=2, le=64)
    gradient_evaluations: int = Field(ge=2, le=MAX_IG_COMPUTE_STEPS)
    input_target_value: float = Field(allow_inf_nan=False)
    baseline_target_value: float = Field(allow_inf_nan=False)
    values: list[float]
    completeness_delta: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def finite_values(self) -> Self:
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("attribution values must be finite")
        return self


class RepresentationSignal(ContractModel):
    pooling: Literal["document_mean_last_hidden_state"]
    values: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def finite_values(self) -> Self:
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("representation values must be finite")
        return self


class SampleScores(ContractModel):
    sample_id: Identifier
    predicted_candidate_id: Identifier
    scores: list[CandidateScore] = Field(min_length=2, max_length=64)
    probability_temperature: float = Field(gt=0.0, allow_inf_nan=False)
    target_value: float = Field(allow_inf_nan=False)
    tokens: TokenSignal
    attribution: AttributionSignal | None = None
    representation: RepresentationSignal | None = None

    @model_validator(mode="after")
    def validate_attribution(self) -> Self:
        if self.attribution is not None:
            signal = self.attribution
            if len(signal.values) != self.tokens.token_width:
                raise ValueError("attribution values must match token_width")
            start = self.tokens.state_token_start
            end = start + self.tokens.state_token_count
            if any(value != 0.0 for value in signal.values[:start] + signal.values[end:]):
                raise ValueError("document baseline requires zero attribution outside document")
            delta = signal.input_target_value - signal.baseline_target_value
            residual = delta - math.fsum(signal.values[start:end])
            if not math.isclose(residual, signal.completeness_delta, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("saved quantities do not reproduce completeness residual")
            if not math.isclose(signal.input_target_value, self.target_value, abs_tol=1e-6):
                raise ValueError("IG input quantity differs from reported target")
        return self


class ExtractionResponse(ContractModel):
    schema_version: Literal["4"]
    request_id: Identifier
    request_sha256: Fingerprint
    run_id: Identifier
    window_index: int = Field(ge=0)
    model: ModelRevision
    tokenizer: ModelRevision
    runtime: RuntimeIdentity
    decision_target: DecisionTarget
    precision: Literal["float32_no_autocast"]
    tokenization: Literal["laya_sdk_0.3.29_encode_state"]
    pooling: Literal["document_mean_last_hidden_state"]
    score_semantics: Literal["temperature_scaled_choice_probability"]
    batch_context_sha256: Fingerprint
    latency_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    samples: list[SampleScores] = Field(min_length=1, max_length=32)


class Readiness(ContractModel):
    schema_version: Literal["4"] = "4"
    status: Literal["not_ready", "ready"] = "not_ready"
    reason: str | None = "No model adapter has been configured."
    extraction_available: bool = False
    capabilities: list[str] = Field(default_factory=list)
    scientifically_qualified: Literal[False] = False
    model: ModelRevision | None = None
    runtime: RuntimeIdentity | None = None
