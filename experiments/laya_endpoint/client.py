"""Strict client for the remote Laya extraction contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

COMPLETENESS_ATOL = 0.01
COMPLETENESS_RTOL = 0.01
MAX_IG_COMPUTE_STEPS = 16384

MAX_ENDPOINT_REQUEST_SAMPLES = 8


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def split_request(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one cached batch into smaller, independently fingerprinted calls."""
    samples = request["samples"]
    if len(samples) <= MAX_ENDPOINT_REQUEST_SAMPLES:
        return [copy.deepcopy(request)]
    subrequests = []
    for index, start in enumerate(range(0, len(samples), MAX_ENDPOINT_REQUEST_SAMPLES)):
        subrequest = copy.deepcopy(request)
        subrequest["samples"] = samples[start : start + MAX_ENDPOINT_REQUEST_SAMPLES]
        subrequest["request_id"] = f"{request['request_id']}-part{index}"
        subrequest["corpus_sha256"] = sha256_json(subrequest["samples"])
        subrequests.append(subrequest)
    return subrequests


def load_cached_subresponses(path: Path, request: dict[str, Any]) -> list[dict[str, Any]]:
    """Load validated child responses from a partially completed parent batch."""
    if not path.exists():
        return []
    entry = json.loads(path.read_text(encoding="utf-8"))
    if entry.get("request_sha256") != sha256_json(request):
        raise ValueError(f"cached window {path} belongs to a different request")
    subresponses = entry.get("subresponses", [])
    if not isinstance(subresponses, list):
        raise ValueError(f"cached window {path} has invalid subresponses")
    _validate_subresponses(request, subresponses)
    return subresponses


def _validate_subresponses(
    request: dict[str, Any], subresponses: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Accept only exact planned child batches, including text and runtime."""
    expected = {sha256_json(child): child for child in split_request(request)}
    results = {}
    seen = set()
    for record in subresponses:
        if not isinstance(record, dict):
            raise ValueError("cached subresponse must be an object")
        child = record.get("request")
        fingerprint = record.get("request_sha256")
        if (
            not isinstance(child, dict)
            or fingerprint not in expected
            or fingerprint in seen
            or child != expected[fingerprint]
            or sha256_json(child) != fingerprint
        ):
            raise ValueError("cached subrequest differs from exact planned batch context")
        seen.add(fingerprint)
        response = record.get("response")
        if not isinstance(response, dict) or "execution_batches" in response:
            raise ValueError("cached child response is invalid")
        validate_response(child, response)
        if record.get("endpoint_image_digest") != request["runtime"]["configured_image_digest"]:
            raise ValueError("cached subresponse image identity differs from requested runtime")
        for sample in response["samples"]:
            if sample["sample_id"] in results:
                raise ValueError("cached subresponses duplicate a sample")
            results[sample["sample_id"]] = sample
    return results, [
        results[sample["sample_id"]]
        for sample in request["samples"]
        if sample["sample_id"] in results
    ]


def _merged_payload(request, subresponses, ordered_samples):
    response = copy.deepcopy(subresponses[0]["response"])
    response.update(
        request_id=request["request_id"],
        request_sha256=sha256_json(request),
        run_id=request["run_id"],
        window_index=request["window_index"],
        samples=ordered_samples,
        latency_seconds=sum(part["response"]["latency_seconds"] for part in subresponses),
        execution_batches=copy.deepcopy(subresponses),
    )
    # Never claim concatenated child outputs were computed in the parent batch.
    response.pop("batch_context_sha256")
    return response


def merge_subresponses(
    request: dict[str, Any], subresponses: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return a host aggregate retaining every exact executed child batch."""
    results, ordered = _validate_subresponses(request, subresponses)
    if len(results) != len(request["samples"]):
        return None
    return _merged_payload(request, subresponses, ordered)


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def validate_response(request: dict[str, Any], response: dict[str, Any]) -> None:
    """Independently reconstruct scores, fixed targets and completeness."""
    if request.get("schema_version") != "2":
        raise ValueError("schema 1 candidate-logit caches are historical pilot artifacts only")
    if "execution_batches" in response:
        parts = response["execution_batches"]
        if not isinstance(parts, list):
            raise ValueError("host aggregate execution batches are invalid")
        results, ordered = _validate_subresponses(request, parts)
        if len(results) != len(request["samples"]) or response != _merged_payload(
            request, parts, ordered
        ):
            raise ValueError("host aggregate differs from exact executed child batches")
        return
    for field in (
        "schema_version",
        "request_id",
        "run_id",
        "window_index",
        "model",
        "tokenizer",
        "runtime",
        "decision_target",
        "precision",
        "tokenization",
        "pooling",
    ):
        if response.get(field) != request[field]:
            raise ValueError(f"endpoint response {field} does not match the request")
    if response.get("request_sha256") != sha256_json(request):
        raise ValueError("endpoint exact request fingerprint differs")
    if response.get("batch_context_sha256") != sha256_json(request["samples"]):
        raise ValueError("endpoint execution batch context differs")
    if response.get("score_semantics") != "temperature_scaled_choice_probability":
        raise ValueError("endpoint response has unexpected score semantics")
    if not _finite(response.get("latency_seconds")) or response["latency_seconds"] < 0:
        raise ValueError("endpoint response latency is invalid")
    candidate_ids = [candidate["candidate_id"] for candidate in request["candidates"]]
    returned = response.get("samples")
    if not isinstance(returned, list) or len(returned) != len(request["samples"]):
        raise ValueError("endpoint response sample count does not match the request")
    target = request["decision_target"]
    for expected, actual in zip(request["samples"], returned, strict=True):
        sample_id = expected["sample_id"]
        if not isinstance(actual, dict) or actual.get("sample_id") != sample_id:
            raise ValueError(f"endpoint response sample order differs at {sample_id}")
        scores = actual.get("scores")
        if not isinstance(scores, list) or not all(isinstance(score, dict) for score in scores):
            raise ValueError("endpoint scores are invalid")
        if [score.get("candidate_id") for score in scores] != candidate_ids:
            raise ValueError(f"endpoint score order differs for {sample_id}")
        logits = [score.get("logit") for score in scores]
        probabilities = [score.get("probability") for score in scores]
        temperature = actual.get("probability_temperature")
        if (
            not all(_finite(value) for value in logits)
            or not _finite(temperature)
            or temperature <= 0
        ):
            raise ValueError("endpoint logits or temperature are invalid")
        if any(not _finite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError("endpoint probability is invalid")
        if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=2e-6):
            raise ValueError("endpoint probabilities do not sum to one")
        maximum = max(logits)
        exponential = [math.exp((value - maximum) / temperature) for value in logits]
        denominator = math.fsum(exponential)
        if any(
            not math.isclose(p, z / denominator, abs_tol=2e-6, rel_tol=2e-5)
            for p, z in zip(probabilities, exponential, strict=True)
        ):
            raise ValueError("endpoint probabilities differ from saved forward logits")
        predicted = candidate_ids[max(range(len(logits)), key=logits.__getitem__)]
        if actual.get("predicted_candidate_id") != predicted:
            raise ValueError("endpoint prediction differs from saved forward logits")
        margin = (
            logits[candidate_ids.index(target["target_candidate_id"])]
            - logits[candidate_ids.index(target["reference_candidate_id"])]
        )
        if not _finite(actual.get("target_value")) or not math.isclose(
            actual["target_value"], margin, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise ValueError("endpoint fixed target differs from saved forward logits")

        tokens = actual.get("tokens")
        if not isinstance(tokens, dict):
            raise ValueError("endpoint omitted token metadata")
        width = tokens.get("token_width")
        if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 8192:
            raise ValueError("endpoint token width is invalid")
        for field in ("token_ids", "tokens", "segments"):
            if not isinstance(tokens.get(field), list) or len(tokens[field]) != width:
                raise ValueError("endpoint token arrays are misaligned")
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in tokens["token_ids"]
        ):
            raise ValueError("endpoint token IDs are invalid")
        if any(
            not isinstance(value, str)
            for field in ("tokens", "segments")
            for value in tokens[field]
        ):
            raise ValueError("endpoint token strings or segments are invalid")
        counts = [
            tokens.get(field)
            for field in (
                "state_token_start",
                "state_token_count",
                "state_tokens_total",
                "state_tokens_dropped",
            )
        ]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in counts):
            raise ValueError("endpoint token counts are invalid")
        start, count, total, dropped = counts
        end = start + count
        if start < 0 or count < 0 or end > width or total < count or dropped != total - count:
            raise ValueError("endpoint token span or truncation count is invalid")
        if [i for i, segment in enumerate(tokens["segments"]) if segment == "document"] != list(
            range(start, end)
        ):
            raise ValueError("endpoint document span is misaligned")
        representation = actual.get("representation")
        if request.get("representations", False):
            if (
                not isinstance(representation, dict)
                or representation.get("pooling") != request["pooling"]
                or count == 0
                or not isinstance(representation.get("values"), list)
                or not representation["values"]
                or not all(_finite(v) for v in representation["values"])
            ):
                raise ValueError("endpoint requested document representation is invalid")
        elif representation is not None:
            raise ValueError("endpoint returned an unrequested representation")

        signal = actual.get("attribution")
        plan = request.get("attribution")
        if plan is None:
            if signal is not None:
                raise ValueError("endpoint returned an unrequested attribution")
            continue
        if not isinstance(signal, dict) or any(
            signal.get(key) != value for key, value in target.items()
        ):
            raise ValueError("endpoint attribution target differs")
        if (
            signal.get("method") != plan["method"]
            or signal.get("baseline") != "pad_document_tokens"
        ):
            raise ValueError("endpoint attribution function or baseline differs")
        evaluations = signal.get("gradient_evaluations")
        if (
            not isinstance(evaluations, int)
            or isinstance(evaluations, bool)
            or not 2 <= evaluations <= MAX_IG_COMPUTE_STEPS
            or signal.get("n_steps_requested") != plan.get("n_steps", 64)
        ):
            raise ValueError("endpoint IG evaluation count is invalid")
        values = signal.get("values")
        if not isinstance(values, list) or len(values) != width:
            raise ValueError("endpoint attribution arrays are misaligned")
        if not all(_finite(value) for value in values):
            raise ValueError("endpoint attribution contains non-finite values")
        if any(value != 0.0 for value in values[:start] + values[end:]):
            raise ValueError("endpoint attribution changed non-document baseline positions")
        input_value = signal.get("input_target_value")
        baseline_value = signal.get("baseline_target_value")
        residual = signal.get("completeness_delta")
        if not all(_finite(value) for value in (input_value, baseline_value, residual)):
            raise ValueError("endpoint saved completeness quantities are invalid")
        if not math.isclose(input_value, actual["target_value"], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("endpoint IG input and scoring forward quantities differ")
        delta = input_value - baseline_value
        recalculated = delta - math.fsum(values[start:end])
        if not math.isclose(residual, recalculated, abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError("endpoint saved quantities do not reproduce completeness residual")
        tolerance = COMPLETENESS_ATOL + COMPLETENESS_RTOL * abs(delta)
        if abs(recalculated) > tolerance + 1e-6:
            raise ValueError("endpoint completeness residual exceeds the implementation gate")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist one complete JSON record without leaving partial cache files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


class EndpointClient:
    """Send endpoint batches and verify every returned signal before use."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout: float,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self.headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Laya endpoint returned HTTP {error.code}: {detail}") from error

    def health(self) -> dict[str, Any]:
        payload = self._request("/health", timeout=60.0)
        if payload.get("status") != "ready":
            raise RuntimeError(f"Laya endpoint is not ready: {payload}")
        return payload

    def extract(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = self._request(
            "/extract",
            method="POST",
            payload=request,
            timeout=self.timeout,
        )
        validate_response(request, payload)
        return payload


def load_cached_response(path: Path, request: dict[str, Any]) -> dict[str, Any] | None:
    """Load only a cache entry produced for the exact canonical request."""
    if not path.exists():
        return None
    entry = json.loads(path.read_text(encoding="utf-8"))
    expected_fingerprint = sha256_json(request)
    if entry.get("request_sha256") != expected_fingerprint:
        raise ValueError(f"cached window {path} belongs to a different request")
    if "subresponses" in entry:
        return merge_subresponses(request, load_cached_subresponses(path, request))
    response = entry.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"cached window {path} has no response object")
    validate_response(request, response)
    return response
