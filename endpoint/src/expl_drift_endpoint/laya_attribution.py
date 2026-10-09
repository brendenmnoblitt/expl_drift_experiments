"""Integrated Gradients on Laya's input token embeddings and decision logits."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from expl_drift_endpoint.contract import AttributionSignal

COMPLETENESS_ATOL = 0.01
COMPLETENESS_RTOL = 0.01


@lru_cache(maxsize=8)
def _gauss_legendre_rule(n_steps: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    nodes, weights = np.polynomial.legendre.leggauss(n_steps)
    return tuple(float(node) for node in nodes), tuple(float(weight) for weight in weights)


def integrate_gradients(
    inputs: Any,
    baselines: Any,
    gradient_at: Callable[[Any], Any],
    n_steps: int,
) -> Any:
    """Gauss-Legendre Integrated Gradients for matching arrays or tensors."""
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if inputs.shape != baselines.shape:
        raise ValueError("inputs and baselines must have the same shape")

    delta = inputs - baselines
    nodes, weights = _gauss_legendre_rule(n_steps)
    gradient_sum: Any = None
    for node, weight in zip(nodes, weights, strict=True):
        point = baselines + delta * ((node + 1.0) / 2.0)
        gradient = gradient_at(point)
        if gradient.shape != inputs.shape:
            raise ValueError("gradient_at must return a gradient with the input shape")
        weighted_gradient = gradient * (weight / 2.0)
        gradient_sum = (
            weighted_gradient if gradient_sum is None else gradient_sum + weighted_gradient
        )
    return delta * gradient_sum


def _validate_completeness(completeness_delta: float, logit_delta: float) -> None:
    tolerance = COMPLETENESS_ATOL + COMPLETENESS_RTOL * abs(logit_delta)
    if abs(completeness_delta) > tolerance:
        raise RuntimeError(
            "Integrated Gradients completeness residual "
            f"{completeness_delta:.6g} exceeds tolerance {tolerance:.6g}"
        )


def state_token_span(sequence_length: int, state_stats: Mapping[str, int]) -> tuple[int, int]:
    """Locate the retained document tokens before Laya's closing separator."""
    used = state_stats["state_tokens_used"]
    if used < 0 or used > sequence_length - 1:
        raise ValueError("Laya state-token count is incompatible with its encoded sequence")
    end = sequence_length - 1
    return end - used, end


def _model_inputs(agent: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.to(agent.device)
        for key, value in batch.items()
        if key
        in (
            "input_ids",
            "attention_mask",
            "marker_pos",
            "marker_mask",
            "qtype",
            "position_ids",
            "option_ids",
        )
    }


def _call_with_embeddings(agent: Any, batch: Mapping[str, Any], embeddings: Any, target_index: int):
    """Run the unchanged Laya decision head with supplied word embeddings."""
    torch = __import__("torch")
    model = agent.model
    device = agent.device
    values = _model_inputs(agent, batch)

    def substitute_embeddings(module, args, kwargs):
        kwargs = dict(kwargs)
        kwargs["input_ids"] = None
        kwargs["inputs_embeds"] = embeddings
        return args, kwargs

    embedding_module = model.encoder.embeddings
    handle = embedding_module.register_forward_pre_hook(substitute_embeddings, with_kwargs=True)
    try:
        with torch.autocast(device_type=device.type, dtype=agent.dtype, enabled=agent.amp_enabled):
            logits, _ = model(**values)
        return logits[0, target_index]
    finally:
        handle.remove()


def explain_choice(
    agent: Any,
    state: str,
    instructions: str,
    candidates: Sequence[tuple[str, str]],
    target_candidate_id: str,
    *,
    max_tokens: int,
    n_steps: int,
) -> AttributionSignal:
    """Attribute one Laya candidate logit to retained document-token embeddings.

    The SDK's private sequence builder and collator preserve the exact serialization,
    candidate markers, and truncation used by ``predict_batch``. A forward hook only
    substitutes ModernBERT token embeddings; Laya's encoder and decision head still run.
    """
    import torch
    from laya.common import collate_items

    candidate_ids = [candidate_id for candidate_id, _ in candidates]
    target_index = candidate_ids.index(target_candidate_id)
    question = {
        "type": "choice",
        "instructions": instructions,
        "criteria": {candidate_id: description for candidate_id, description in candidates},
    }
    internal = {"expl_drift_target": agent._to_internal(question)}
    item = agent._encode_state(
        state,
        ["expl_drift_target"],
        internal,
        max_len=max_tokens,
        head_max_len=int(agent.cfg.get("head_max_len", 192)),
    )[0]
    pad_id = agent.tok.pad_token_id
    if pad_id is None:
        raise ValueError("Laya tokenizer must define a pad token for the IG baseline")
    batch = collate_items([[item]], pad_id)
    input_ids = batch["input_ids"].to(agent.device)
    state_start, state_end = state_token_span(len(item["ids"]), item["state_stats"])
    state_tokens_used = state_end - state_start

    baseline_ids = input_ids.clone()
    baseline_ids[:, state_start:state_end] = pad_id
    word_embeddings = agent.model.encoder.get_input_embeddings()
    with torch.no_grad():
        input_embeds = word_embeddings(input_ids).detach()
        baseline_embeds = word_embeddings(baseline_ids).detach()

    # Prove the embedding-input hook preserves the original decision logit first.
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=agent.device.type,
            dtype=agent.dtype,
            enabled=agent.amp_enabled,
        ),
    ):
        direct_logits, _ = agent.model(**_model_inputs(agent, batch))
        input_logit = _call_with_embeddings(agent, batch, input_embeds, target_index)
        baseline_logit = _call_with_embeddings(agent, batch, baseline_embeds, target_index)
    if not torch.allclose(input_logit, direct_logits[0, target_index], rtol=1e-4, atol=1e-4):
        raise RuntimeError("Laya embedding path changed the target decision logit")
    input_logit_value = float(input_logit.float().cpu().item())
    baseline_logit_value = float(baseline_logit.float().cpu().item())

    def gradient_at(point):
        point = point.detach().requires_grad_(True)
        with torch.enable_grad():
            score = _call_with_embeddings(agent, batch, point, target_index)
            return torch.autograd.grad(score, point, retain_graph=False)[0].detach()

    token_attributions = integrate_gradients(input_embeds, baseline_embeds, gradient_at, n_steps)
    if not torch.isfinite(token_attributions).all():
        raise RuntimeError("Laya Integrated Gradients produced non-finite values")
    token_values = token_attributions.sum(dim=-1)[0].detach().float().cpu().tolist()
    state_values = [float(value) for value in token_values[state_start:state_end]]
    logit_delta = input_logit_value - baseline_logit_value
    completeness_delta = logit_delta - sum(state_values)
    _validate_completeness(completeness_delta, logit_delta)

    sequence_ids = list(item["ids"])
    sequence_tokens = agent.tok.convert_ids_to_tokens(sequence_ids)
    sequence_segments = ["instruction"] * len(sequence_ids)
    markers = item["markers"]
    for index, (candidate_id, _) in enumerate(candidates):
        start = markers[index]
        end = markers[index + 1] if index + 1 < len(markers) else state_start - 1
        for position in range(start, end):
            sequence_segments[position] = f"candidate:{candidate_id}"
        sequence_segments[start] = f"candidate_marker:{candidate_id}"
    sequence_segments[state_start:state_end] = ["document"] * state_tokens_used
    special_ids = set(agent.tok.all_special_ids)
    for position, token_id in enumerate(sequence_ids):
        if token_id in special_ids and sequence_segments[position] == "instruction":
            sequence_segments[position] = "special"

    state_stats = item["state_stats"]
    total_tokens = int(state_stats["state_tokens"])
    dropped_tokens = int(state_stats["state_tokens_dropped"])
    width = int(agent.cfg.get("max_len", max_tokens))
    padding = width - len(sequence_ids)
    if padding < 0:
        raise ValueError("Laya encoded sequence exceeds the configured model width")
    sequence_ids.extend([pad_id] * padding)
    sequence_tokens.extend(agent.tok.convert_ids_to_tokens([pad_id] * padding))
    sequence_segments.extend(["padding"] * padding)
    token_values.extend([0.0] * padding)

    return AttributionSignal(
        method="integrated_gradients",
        target_candidate_id=target_candidate_id,
        target_semantics="candidate_logit",
        baseline="pad_document_tokens",
        n_steps=n_steps,
        token_width=width,
        token_ids=sequence_ids,
        tokens=sequence_tokens,
        segments=sequence_segments,
        values=[float(value) for value in token_values],
        state_token_start=state_start,
        state_token_count=state_tokens_used,
        state_tokens_total=total_tokens,
        state_tokens_dropped=dropped_tokens,
        completeness_delta=completeness_delta,
    )
