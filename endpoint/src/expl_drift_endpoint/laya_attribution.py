"""Integrated Gradients on Laya's input token embeddings and decision logits."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from expl_drift_endpoint.contract import (
    MAX_IG_COMPUTE_STEPS,
    AttributionSignal,
    TokenSignal,
)

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


def _forward(agent: Any, batch: Mapping[str, Any], embeddings: Any = None, *, capture=False):
    """Run the actual SDK DecisionModel, disabling autocast for every quantity.

    laya 0.3.29 common.DecisionModel calls encoder(...).last_hidden_state before
    its type embedding/head. Capture precisely that tensor, not the head's CLS pool.
    ModernBERT's embeddings module accepts input_ids and inputs_embeds; replacing
    only that module leaves SDK attention/option masks and decision heads intact.
    """
    import torch

    handles = []
    hidden = []
    if embeddings is not None:
        def substitute_embeddings(module, args, kwargs):
            kwargs = dict(kwargs)
            kwargs["input_ids"] = None
            kwargs["inputs_embeds"] = embeddings
            return args, kwargs

        handles.append(agent.model.encoder.embeddings.register_forward_pre_hook(
            substitute_embeddings, with_kwargs=True
        ))
    if capture:
        def save_hidden(module, args, output):
            hidden.append(output.last_hidden_state)

        handles.append(agent.model.encoder.register_forward_hook(save_hidden))
    try:
        with torch.autocast(device_type=agent.device.type, enabled=False):
            logits, _ = agent.model(**_model_inputs(agent, batch))
        if capture and len(hidden) != 1:
            raise RuntimeError("SDK encoder must emit exactly one last_hidden_state")
        return logits, hidden[0] if capture else None
    finally:
        for handle in handles:
            handle.remove()


def encode_samples(agent: Any, request: Any):
    """Encode request samples, then collate each in the fixed singleton context."""
    from laya.common import collate_items

    question = {
        "type": "choice",
        "instructions": request.instructions,
        "criteria": {
            candidate.candidate_id: candidate.description for candidate in request.candidates
        },
    }
    internal = {"expl_drift_target": agent._to_internal(question)}
    items = [
        agent._encode_state(
            sample.text,
            ["expl_drift_target"],
            internal,
            max_len=request.max_tokens,
            head_max_len=int(agent.cfg.get("head_max_len", 192)),
        )[0]
        for sample in request.samples
    ]
    pad_id = agent.tok.pad_token_id
    if pad_id is None:
        raise ValueError("Laya tokenizer must define a pad token for the document baseline")
    batches = [collate_items([[item]], pad_id) for item in items]
    return items, batches


def token_signal(agent: Any, item: Mapping[str, Any], candidate_ids: Sequence[str]) -> TokenSignal:
    sequence_ids = list(item["ids"])
    start, end = state_token_span(len(sequence_ids), item["state_stats"])
    segments = ["instruction"] * len(sequence_ids)
    markers = item["markers"]
    if len(markers) != len(candidate_ids):
        raise ValueError("SDK candidate markers differ from requested candidates")
    for index, candidate_id in enumerate(candidate_ids):
        candidate_start = markers[index]
        candidate_end = markers[index + 1] if index + 1 < len(markers) else start - 1
        segments[candidate_start:candidate_end] = [
            f"candidate:{candidate_id}"
        ] * (candidate_end - candidate_start)
        segments[candidate_start] = f"candidate_marker:{candidate_id}"
    segments[start:end] = ["document"] * (end - start)
    special_ids = set(agent.tok.all_special_ids)
    for position, token_id in enumerate(sequence_ids):
        if token_id in special_ids and segments[position] == "instruction":
            segments[position] = "special"
    width = int(agent.cfg["max_len"])
    padding = width - len(sequence_ids)
    if padding < 0:
        raise ValueError("SDK sequence exceeds the configured model width")
    sequence_ids.extend([agent.tok.pad_token_id] * padding)
    segments.extend(["padding"] * padding)
    return TokenSignal(
        token_width=width, token_ids=sequence_ids,
        tokens=agent.tok.convert_ids_to_tokens(sequence_ids), segments=segments,
        state_token_start=start, state_token_count=end - start,
        state_tokens_total=int(item["state_stats"]["state_tokens"]),
        state_tokens_dropped=int(item["state_stats"]["state_tokens_dropped"]),
    )


def document_mean(hidden: Any, row: int, tokens: TokenSignal):
    """Exclude instruction/candidate/separator/padding tokens from encoder pooling."""
    start = tokens.state_token_start
    end = start + tokens.state_token_count
    if end == start:
        raise ValueError("document mean pooling requires at least one retained document token")
    if hidden.ndim != 3 or end > hidden.shape[1]:
        raise ValueError("encoder last_hidden_state is incompatible with the document span")
    return hidden[row, start:end].mean(dim=0)


def explain_margin(
    agent: Any,
    batch: Mapping[str, Any],
    tokens: TokenSignal,
    target: Any,
    candidate_ids: Sequence[str],
    direct_logits: Any,
    *,
    n_steps: int,
) -> AttributionSignal:
    """Explain one sample's fixed margin in the singleton scoring context."""
    import torch

    target_index = candidate_ids.index(target.target_candidate_id)
    reference_index = candidate_ids.index(target.reference_candidate_id)
    input_ids = batch["input_ids"].to(agent.device)
    if input_ids.shape[0] != 1:
        raise ValueError("integrated gradients requires a singleton model context")
    start, end = tokens.state_token_start, tokens.state_token_start + tokens.state_token_count
    baseline_ids = input_ids.clone()
    baseline_ids[0, start:end] = agent.tok.pad_token_id
    word_embeddings = agent.model.encoder.get_input_embeddings()
    with torch.no_grad():
        input_embeds = word_embeddings(input_ids).detach()
        baseline_embeds = word_embeddings(baseline_ids).detach()
        input_logits, _ = _forward(agent, batch, input_embeds)
        baseline_logits, _ = _forward(agent, batch, baseline_embeds)
    if not torch.allclose(input_logits, direct_logits, rtol=1e-4, atol=1e-4):
        raise RuntimeError("Laya embedding path changed the reported decision logits")

    def quantity(logits):
        return logits[0, target_index] - logits[0, reference_index]

    input_value = float(quantity(input_logits).cpu().item())
    baseline_value = float(quantity(baseline_logits).cpu().item())

    def gradient_at(point):
        point = point.detach().requires_grad_(True)
        with torch.enable_grad():
            logits, _ = _forward(agent, batch, point)
            return torch.autograd.grad(quantity(logits), point, retain_graph=False)[0].detach()

    target_delta = input_value - baseline_value
    attributions, evaluations = integrate_gradients(
        input_embeds, baseline_embeds, gradient_at, n_steps,
        max_steps=MAX_IG_COMPUTE_STEPS, target_delta=target_delta,
        completeness_tolerance=_completeness_tolerance(target_delta),
    )
    if not torch.isfinite(attributions).all():
        raise RuntimeError("Laya Integrated Gradients produced non-finite values")
    values = attributions.sum(dim=-1)[0].detach().cpu().tolist()
    values.extend([0.0] * (tokens.token_width - len(values)))
    import math

    residual = target_delta - math.fsum(values[start:end])
    _validate_completeness(residual, target_delta)
    return AttributionSignal(
        method="integrated_gradients", target_semantics="fixed_candidate_margin",
        target_candidate_id=target.target_candidate_id,
        reference_candidate_id=target.reference_candidate_id,
        baseline="pad_document_tokens", n_steps_requested=n_steps,
        gradient_evaluations=evaluations, input_target_value=input_value,
        baseline_target_value=baseline_value, values=values, completeness_delta=residual,
    )
