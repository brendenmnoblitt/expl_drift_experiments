# Extraction endpoint scaffold

This standalone service establishes the HTTP input boundary for classifier
experiments. It has no model adapter yet: `/live` reports process liveness,
`/health` returns HTTP 503 with `extraction_available: false`, and a valid
`/extract` request returns HTTP 503. Invalid requests return HTTP 422.
It must not be deployed as a ready inference endpoint in this state.

The endpoint package runs independently of the experiments package's eager
training/plotting imports. It performs no drift calculations. The experiment
runner will pass saved signals through a pinned `expl_drift` revision.

## Local checks

From this directory, use Python 3.12 and the committed dependency lock:

```bash
rtk uv sync --locked
rtk uv run --locked pytest
rtk uv run --locked ruff check src tests scripts
rtk uv run --locked ruff format --check src tests scripts
rtk uv run --locked uvicorn expl_drift_endpoint.app:app --host 127.0.0.1 --port 8000
```

The tests exercise readiness, strict validation, immutable revision requirements,
sample/candidate identity, fixed attribution targets, and text preservation.
They use synthetic revision fixtures and make no model calls.

## Container checks

Build from this directory so the context excludes legacy data and artifacts:

```bash
rtk proxy docker build --platform linux/amd64 -t expl-drift-endpoint:scaffold .
rtk proxy docker run --rm -p 127.0.0.1:8000:8000 expl-drift-endpoint:scaffold
```

This first image is a CPU service scaffold with a pinned base-image digest and
locked runtime dependencies. GPU dependencies, model loading, HF deployment,
and numerical extraction parity remain subsequent increments. HF model artifacts
will be mounted at `/repository`; the scaffold does not read that mount.

## CI checks

The workflow at `.github/workflows/endpoint-ci.yml` runs when endpoint or workflow
files change on a `codex/` branch push or a pull request. It installs the locked
endpoint environment, runs Ruff and the contract suite, builds Linux amd64, and
starts the image using its default command for an HTTP smoke check. The workflow
has read-only repository permissions and requires no HF or registry credentials.
It does not publish images, deploy endpoints, create releases, or merge branches.

With the local container above running, run the same smoke check in a second
terminal from this directory:

```bash
rtk proxy python3 scripts/smoke_container.py
```

The check waits at most approximately 31 seconds for liveness, then verifies
`/health` and valid `/extract` requests return 503 and mutable model revisions
return 422. Its synthetic request loads no model. A qualified model adapter will
need a different readiness/inference smoke check later.

GitHub execution starts after these files are committed and pushed to the
development branch. Local qualification alone does not confirm a hosted CI run.

## Input contract

Schema version `1` accepts one instruction and ordered candidate set for a batch
of samples. Each request identifies its run/window and corpus SHA-256, and pins
model/tokenizer snapshots by full commit SHA. A separate head snapshot can be
supplied for CLM. Optional attributions require an explicit candidate target.

Input text is retained exactly. Duplicate sample/candidate IDs, unknown fields,
mutable revisions, unknown attribution targets, and implicit scalar conversions
are rejected. Current transport bounds are 32 samples, 64 candidates, and 8192
requested tokens; these are safety limits, not claims about model capacity.
The adapter must later validate model-specific budgets and actual loaded pins.

Successful signal responses, feature representations, token mappings, and
extraction capability reporting will be defined with the first qualified model
adapter. Neither attention nor integrated gradients is advertised as supported
by this scaffold.

## Qualification on October 3 2026

- Contract suite: 13 passed using the isolated Python 3.12.13 environment.
- Ruff checks and formatting: passed for the new source and tests.
- Linux amd64 image: built successfully, using Python 3.12.15 from the pinned base.
- Real Uvicorn smoke: `/live` returned 200, `/health` and valid `/extract`
  requests returned 503, and a mutable model revision returned 422.

The test client emits one upstream Starlette/AnyIO deprecation warning. The
tests still pass; it does not affect the production Uvicorn startup check.
These results qualify the service boundary, not classifier inference or drift
calculations. No model weights were downloaded and no HF endpoint was deployed.
