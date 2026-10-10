"""Run fixed examples through the pinned Laya SDK on HF job hardware."""

from __future__ import annotations

import json
import math

import torch
from huggingface_hub import snapshot_download
from laya import Agent

MODEL_ID = "convaiinnovations/laya"
MODEL_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
QUESTION_ID = "expl_drift_target"
MAX_TOKENS = 128
INSTRUCTIONS = "Select the candidate that best describes the sentiment expressed by the document."
CANDIDATES = [
    ("negative", "The document expresses a negative sentiment."),
    ("positive", "The document expresses a positive sentiment."),
]
SAMPLES = [
    (
        "short-pos",
        "I loved the whole experience: quick service, thoughtful staff, and excellent food.",
    ),
    (
        "short-neg",
        "The order arrived late, the meal was cold, and the staff ignored our complaint.",
    ),
    (
        "medium-mix",
        "The product itself was useful, but delivery was late and customer support never replied.",
    ),
    (
        "long-truncated",
        "The customer reports late delivery, a cold meal, and an ignored complaint. "
        "The experience was disappointing and they would not recommend the service. " * 180,
    ),
]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Run this reference on HF GPU job hardware, not the host.")

    model_path = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    agent = Agent(model_id_or_path=model_path, device="cuda")
    questions = {
        QUESTION_ID: {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": dict(CANDIDATES),
        }
    }
    samples = []
    for sample_id, text in SAMPLES:
        result = agent.predict_batch([text], questions, batch_size=1, max_len=MAX_TOKENS)[0]
        answer = result["answers"][QUESTION_ID]
        probabilities = answer["probabilities"]
        ordered = [float(probabilities[candidate_id]) for candidate_id, _ in CANDIDATES]
        if set(probabilities) != {candidate_id for candidate_id, _ in CANDIDATES}:
            raise RuntimeError(f"Native SDK candidate IDs differ for {sample_id}")
        if any(not math.isfinite(value) for value in ordered):
            raise RuntimeError(f"Native SDK returned non-finite scores for {sample_id}")
        samples.append(
            {
                "sample_id": sample_id,
                "predicted_candidate_id": answer["choice"],
                "candidate_ids": [candidate_id for candidate_id, _ in CANDIDATES],
                "probabilities": ordered,
            }
        )

    print(
        json.dumps(
            {
                "model": {"repository": MODEL_ID, "revision": MODEL_REVISION},
                "max_tokens": MAX_TOKENS,
                "samples": samples,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
