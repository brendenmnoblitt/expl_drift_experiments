"""Check the running scaffold container at localhost:8000 using only the stdlib."""

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(path: str, payload: dict | None = None) -> tuple[int, dict]:
    """Return JSON from both successful and rejected HTTP requests."""
    data = None if payload is None else json.dumps(payload).encode()
    message = Request(
        f"http://127.0.0.1:8000{path}", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        response = urlopen(message, timeout=1)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def main() -> None:
    deadline = time.monotonic() + 30
    while True:
        try:
            status, body = request("/live")
            break
        except (URLError, TimeoutError, ConnectionError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Container did not become live within 30 seconds") from None
            time.sleep(0.25)
    if status != 200 or body != {"status": "alive"}:
        raise RuntimeError(f"Unexpected liveness response: {status}, {body}")

    status, body = request("/health")
    if status != 503 or body.get("extraction_available") is not False:
        raise RuntimeError(f"Scaffold reported unexpected readiness: {status}, {body}")

    # Synthetic revisions exercise the contract without loading a model.
    revision = {"repository": "fixture/model", "revision": "a" * 40}
    payload = {
        "schema_version": "1",
        "request_id": "container-smoke",
        "run_id": "ci-smoke",
        "window_index": 0,
        "corpus_sha256": "b" * 64,
        "model": revision,
        "tokenizer": revision,
        "instructions": "Choose the department.",
        "candidates": [
            {"candidate_id": "billing", "description": "Invoices"},
            {"candidate_id": "technical", "description": "Outages"},
        ],
        "samples": [{"sample_id": "ticket-1", "text": "I was charged twice."}],
        "max_tokens": 512,
    }
    status, body = request("/extract", payload)
    if status != 503 or body.get("detail", {}).get("request_id") != "container-smoke":
        raise RuntimeError(f"Unexpected valid extraction response: {status}, {body}")

    payload["model"] = {"repository": "fixture/model", "revision": "main"}
    status, body = request("/extract", payload)
    if status != 422:
        raise RuntimeError(f"Mutable model revision was not rejected: {status}, {body}")
    print("Container smoke passed: live 200, health/extract 503, mutable revision 422")


if __name__ == "__main__":
    main()
