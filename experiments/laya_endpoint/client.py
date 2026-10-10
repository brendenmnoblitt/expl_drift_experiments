"""Strict client for the remote Laya extraction contract."""

from __future__ import annotations

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


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def validate_response(request: dict[str, Any], response: dict[str, Any]) -> None:
    """Reject mismatched provenance, scores, or unaligned attributions."""
    for field in ("schema_version", "request_id", "run_id", "window_index"):
        if response.get(field) != request[field]:
            raise ValueError(f"endpoint response {field} does not match the request")
    for field in ("model", "tokenizer"):
        if response.get(field) != request[field]:
            raise ValueError(f"endpoint response {field} revision does not match the request")
    if response.get("score_semantics") != "choice_probability":
        raise ValueError("endpoint response has unexpected score semantics")

    candidate_ids = [candidate["candidate_id"] for candidate in request["candidates"]]
    requested_samples = request["samples"]
    returned_samples = response.get("samples")
    if not isinstance(returned_samples, list) or len(returned_samples) != len(requested_samples):
        raise ValueError("endpoint response sample count does not match the request")

    attribution_plan = request.get("attribution")
    for expected, actual in zip(requested_samples, returned_samples, strict=True):
        sample_id = expected["sample_id"]
        if actual.get("sample_id") != sample_id:
            raise ValueError(f"endpoint response sample order differs at {sample_id}")
        scores = actual.get("scores")
        if not isinstance(scores, list):
            raise ValueError(f"endpoint scores are invalid for {sample_id}")
        if [score.get("candidate_id") for score in scores] != candidate_ids:
            raise ValueError(f"endpoint score order differs for {sample_id}")
        probabilities = [score.get("probability") for score in scores]
        if any(
            not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in probabilities
        ):
            raise ValueError(f"endpoint returned an invalid probability for {sample_id}")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=0.0001 * len(scores)):
            raise ValueError(f"endpoint probabilities do not sum to one for {sample_id}")
        if actual.get("predicted_candidate_id") not in candidate_ids:
            raise ValueError(f"endpoint prediction is not a supplied candidate for {sample_id}")

        attribution = actual.get("attribution")
        if attribution_plan is None:
            if attribution is not None:
                raise ValueError(f"unexpected endpoint attribution for {sample_id}")
            continue
        if not isinstance(attribution, dict):
            raise ValueError(f"endpoint omitted requested attribution for {sample_id}")
        if (
            attribution.get("method") != attribution_plan["method"]
            or attribution.get("target_candidate_id") != attribution_plan["target_candidate_id"]
            or attribution.get("target_semantics") != "candidate_logit"
        ):
            raise ValueError(f"endpoint attribution target differs for {sample_id}")

        width = attribution.get("token_width")
        aligned = ("token_ids", "tokens", "segments", "values")
        if not isinstance(width, int) or any(
            not isinstance(attribution.get(field), list) or len(attribution[field]) != width
            for field in aligned
        ):
            raise ValueError(f"endpoint attribution arrays are misaligned for {sample_id}")
        values = attribution["values"]
        if any(not isinstance(value, int | float) or not math.isfinite(value) for value in values):
            raise ValueError(f"endpoint attribution contains a non-finite value for {sample_id}")
        completeness_delta = attribution.get("completeness_delta")
        if not isinstance(completeness_delta, int | float) or not math.isfinite(completeness_delta):
            raise ValueError(f"endpoint completeness residual is invalid for {sample_id}")

        start = attribution.get("state_token_start")
        count = attribution.get("state_token_count")
        total = attribution.get("state_tokens_total")
        dropped = attribution.get("state_tokens_dropped")
        if (
            not isinstance(start, int)
            or not isinstance(count, int)
            or not isinstance(total, int)
            or not isinstance(dropped, int)
        ):
            raise ValueError(f"endpoint token counts are invalid for {sample_id}")
        end = start + count
        if start < 0 or count < 0 or end > width or total < count or dropped != total - count:
            raise ValueError(f"endpoint token span or truncation count is invalid for {sample_id}")
        if attribution["segments"][start:end] != ["document"] * count:
            raise ValueError(f"endpoint document span is misaligned for {sample_id}")
        estimated_logit_delta = sum(values[start:end]) + completeness_delta
        tolerance = COMPLETENESS_ATOL + COMPLETENESS_RTOL * abs(estimated_logit_delta)
        if abs(completeness_delta) > tolerance + 1e-6:
            raise ValueError(
                f"endpoint completeness residual exceeds the qualified gate for {sample_id}"
            )


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
    response = entry.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"cached window {path} has no response object")
    validate_response(request, response)
    return response
