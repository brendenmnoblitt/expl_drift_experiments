"""Integrated Gradients on Laya's input token embeddings and decision logits."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from expl_drift_endpoint.contract import MAX_IG_COMPUTE_STEPS, AttributionSignal

COMPLETENESS_ATOL = 0.01
COMPLETENESS_RTOL = 0.01
GAUSS_LEGENDRE_PANEL_ORDER = 16


def integrate_gradients(
    inputs: Any,
    baselines: Any,
    gradient_at: Callable[[Any], Any],
    n_steps: int,
    *,
    max_steps: int = MAX_IG_COMPUTE_STEPS,
    target_delta: float | None = None,
    completeness_tolerance: float | None = None,
) -> tuple[Any, int]:
    """Integrate gradients with composite Gauss-Legendre panels."""
    import numpy as np

    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if inputs.shape != baselines.shape:
        raise ValueError("inputs and baselines must have the same shape")
    if max_steps < 2:
        raise ValueError("max_steps must allow at least two gradient evaluations")
    if (target_delta is None) != (completeness_tolerance is None):
        raise ValueError("target_delta and completeness_tolerance must be supplied together")

    delta = inputs - baselines
    steps = min(n_steps, max_steps)
    order = (
        GAUSS_LEGENDRE_PANEL_ORDER
        if steps >= GAUSS_LEGENDRE_PANEL_ORDER and steps % GAUSS_LEGENDRE_PANEL_ORDER == 0
        else steps
    )
    nodes, weights = np.polynomial.legendre.leggauss(order)
    evaluations = 0

    def integrate_interval(start: float, end: float, rule_nodes, rule_weights):
        nonlocal evaluations
        midpoint = (start + end) / 2.0
        half_width = (end - start) / 2.0
        estimate = None
        for node, weight in zip(rule_nodes, rule_weights, strict=True):
            point = baselines + delta * float(midpoint + half_width * node)
            gradient = gradient_at(point)
            if gradient.shape != inputs.shape:
                raise ValueError("gradient_at must return a gradient with the input shape")
            contribution = gradient * float(weight * half_width)
            estimate = contribution if estimate is None else estimate + contribution
            evaluations += 1
        return estimate

    def projected_value(estimate: Any) -> float:
        value = (delta * estimate).sum()
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().item()
        return float(value)

    if target_delta is None:
        panels_count = steps // order
        integrated_gradient = None
        for panel in range(panels_count):
            estimate = integrate_interval(
                panel / panels_count, (panel + 1) / panels_count, nodes, weights
            )
            integrated_gradient = (
                estimate if integrated_gradient is None else integrated_gradient + estimate
            )
        return delta * integrated_gradient, evaluations

    if max_steps < GAUSS_LEGENDRE_PANEL_ORDER + 8:
        raise ValueError("max_steps is too small for adaptive Gauss-Legendre integration")
    low_nodes, low_weights = np.polynomial.legendre.leggauss(8)
    high_nodes, high_weights = np.polynomial.legendre.leggauss(GAUSS_LEGENDRE_PANEL_ORDER)
    panel_count = max(1, min(n_steps // GAUSS_LEGENDRE_PANEL_ORDER, max_steps // 24))

    def panel_estimate(start: float, end: float) -> dict[str, Any]:
        low = integrate_interval(start, end, low_nodes, low_weights)
        high = integrate_interval(start, end, high_nodes, high_weights)
        error = abs(projected_value(high - low))
        return {"start": start, "end": end, "estimate": high, "error": error}

    panels = [
        panel_estimate(index / panel_count, (index + 1) / panel_count)
        for index in range(panel_count)
    ]
    first_estimate = panels[0]["estimate"]
    integrated_gradient = (
        first_estimate.clone() if hasattr(first_estimate, "clone") else first_estimate.copy()
    )
    for panel in panels[1:]:
        integrated_gradient += panel["estimate"]

    while True:
        residual = target_delta - projected_value(integrated_gradient)
        if abs(residual) <= completeness_tolerance:
            return delta * integrated_gradient, evaluations
        if evaluations + 48 > max_steps:
            raise RuntimeError(
                f"Adaptive Integrated Gradients completeness residual {residual:.6g} "
                f"exceeds tolerance {completeness_tolerance:.6g}"
            )

        index = max(
            range(len(panels)),
            key=lambda i: (panels[i]["error"], panels[i]["end"] - panels[i]["start"]),
        )
        old = panels[index]
        midpoint = (old["start"] + old["end"]) / 2.0
        left = panel_estimate(old["start"], midpoint)
        right = panel_estimate(midpoint, old["end"])
        if hasattr(integrated_gradient, "sub_"):
            integrated_gradient.sub_(old["estimate"]).add_(left["estimate"]).add_(right["estimate"])
        else:
            integrated_gradient += left["estimate"] + right["estimate"] - old["estimate"]
        panels[index : index + 1] = [left, right]


def _completeness_tolerance(logit_delta: float) -> float:
    return COMPLETENESS_ATOL + COMPLETENESS_RTOL * abs(logit_delta)


def _validate_completeness(completeness_delta: float, logit_delta: float) -> None:
    tolerance = _completeness_tolerance(logit_delta)
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


def _call_with_embeddings(
    agent: Any,
    batch: Mapping[str, Any],
    embeddings: Any,
    target_index: int,
    *,
    use_autocast: bool = True,
):
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
        with torch.autocast(
            device_type=device.type,
            dtype=agent.dtype,
            enabled=agent.amp_enabled and use_autocast,
        ):
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

    # Keep input, baseline, and gradients on the same full-precision model path.
    with torch.no_grad():
        direct_logits, _ = agent.model(**_model_inputs(agent, batch))
        input_logit = _call_with_embeddings(
            agent, batch, input_embeds, target_index, use_autocast=False
        )
        baseline_logit = _call_with_embeddings(
            agent, batch, baseline_embeds, target_index, use_autocast=False
        )
    if not torch.allclose(input_logit, direct_logits[0, target_index], rtol=1e-4, atol=1e-4):
        raise RuntimeError("Laya embedding path changed the target decision logit")
    input_logit_value = float(input_logit.float().cpu().item())
    baseline_logit_value = float(baseline_logit.float().cpu().item())

    def gradient_at(point):
        point = point.detach().requires_grad_(True)
        with torch.enable_grad():
            score = _call_with_embeddings(agent, batch, point, target_index, use_autocast=False)
            return torch.autograd.grad(score, point, retain_graph=False)[0].detach()

    logit_delta = input_logit_value - baseline_logit_value
    completeness_tolerance = _completeness_tolerance(logit_delta)
    token_attributions, integration_steps = integrate_gradients(
        input_embeds,
        baseline_embeds,
        gradient_at,
        n_steps,
        max_steps=MAX_IG_COMPUTE_STEPS,
        target_delta=logit_delta,
        completeness_tolerance=completeness_tolerance,
    )
    if not torch.isfinite(token_attributions).all():
        raise RuntimeError("Laya Integrated Gradients produced non-finite values")
    token_values = token_attributions.sum(dim=-1)[0].detach().float().cpu().tolist()
    state_values = [float(value) for value in token_values[state_start:state_end]]
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
        n_steps=integration_steps,
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
