"""Versioned input contract; model-specific signal outputs follow qualification."""

import math
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
Commit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Fingerprint = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ContractModel(BaseModel):
    """Reject unknown fields and implicit scalar conversions at the boundary."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ModelRevision(ContractModel):
    """Identify one HF model, tokenizer, or projection-head snapshot."""

    repository: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")]
    revision: Commit


class Sample(ContractModel):
    """Keep input text intact, with identity supplied by the experiment runner."""

    sample_id: Identifier
    text: str = Field(min_length=1, max_length=32768)


class Candidate(ContractModel):
    """A stable candidate identity and the text the classifier will score."""

    candidate_id: Identifier
    description: str = Field(min_length=1, max_length=2048)


class AttributionPlan(ContractModel):
    """Require an explicit candidate target for attribution comparisons."""

    method: Literal["attention", "integrated_gradients"]
    target_candidate_id: Identifier
    n_steps: int = Field(default=64, ge=2, le=64)


class ExtractionRequest(ContractModel):
    """One batch for a fixed question, candidate set, and model configuration."""

    schema_version: Literal["1"]
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
    attribution: AttributionPlan | None = None
    representations: bool = False

    @model_validator(mode="after")
    def validate_identities(self) -> Self:
        """Protect sample/candidate ordering and target identity before execution."""
        sample_ids = [sample.sample_id for sample in self.samples]
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample IDs must be unique")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique")
        if self.attribution and self.attribution.target_candidate_id not in candidate_ids:
            raise ValueError("attribution target must identify a supplied candidate")
        return self


class CandidateScore(ContractModel):
    """One Laya choice probability, retaining request candidate order in its list."""

    candidate_id: Identifier
    probability: float = Field(ge=0.0, le=1.0)


class AttributionSignal(ContractModel):
    """Fixed-width token attribution with the exact input-token mapping."""

    method: Literal["integrated_gradients"]
    target_candidate_id: Identifier
    target_semantics: Literal["candidate_logit"]
    baseline: Literal["pad_document_tokens"]
    n_steps: int = Field(ge=2, le=64)
    token_width: int = Field(ge=1, le=8192)
    token_ids: list[int]
    tokens: list[str]
    segments: list[str]
    values: list[float]
    state_token_start: int = Field(ge=0)
    state_token_count: int = Field(ge=0)
    state_tokens_total: int = Field(ge=0)
    state_tokens_dropped: int = Field(ge=0)
    completeness_delta: float

    @model_validator(mode="after")
    def validate_token_alignment(self) -> Self:
        """Keep every attribution aligned to its token and segment metadata."""
        lengths = {len(self.token_ids), len(self.tokens), len(self.segments), len(self.values)}
        if lengths != {self.token_width}:
            raise ValueError("token IDs, strings, segments, and values must match token_width")
        if self.state_token_start + self.state_token_count > self.token_width:
            raise ValueError("document token span exceeds token_width")
        if self.state_tokens_dropped != self.state_tokens_total - self.state_token_count:
            raise ValueError("dropped token count must match total minus retained state tokens")
        if not math.isfinite(self.completeness_delta) or any(
            not math.isfinite(value) for value in self.values
        ):
            raise ValueError("attribution values and completeness delta must be finite")
        return self


class SampleScores(ContractModel):
    """Decision scores aligned to one input sample."""

    sample_id: Identifier
    predicted_candidate_id: Identifier
    scores: list[CandidateScore] = Field(min_length=2, max_length=64)
    attribution: AttributionSignal | None = None


class ExtractionResponse(ContractModel):
    """Scores emitted by the qualified Laya decision head."""

    schema_version: Literal["1"]
    request_id: Identifier
    run_id: Identifier
    window_index: int = Field(ge=0)
    model: ModelRevision
    tokenizer: ModelRevision
    score_semantics: Literal["choice_probability"]
    samples: list[SampleScores] = Field(min_length=1, max_length=32)


class Readiness(ContractModel):
    """Report whether the model adapter has loaded and which signal is available."""

    schema_version: Literal["1"] = "1"
    status: Literal["not_ready", "ready"] = "not_ready"
    reason: str | None = "No model adapter has been configured."
    extraction_available: bool = False
    capabilities: list[str] = Field(default_factory=list)
    model: ModelRevision | None = None
