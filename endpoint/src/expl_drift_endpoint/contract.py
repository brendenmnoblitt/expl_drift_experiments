"""Versioned input contract; model-specific signal outputs follow qualification."""

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


class Readiness(ContractModel):
    """Describe the scaffold honestly until a model adapter is qualified."""

    schema_version: Literal["1"] = "1"
    status: Literal["not_ready"] = "not_ready"
    reason: str = "No model adapter has been configured and qualified."
    extraction_available: Literal[False] = False
