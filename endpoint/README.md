# Laya extraction endpoint

The endpoint has a CPU-only validation image and a separate Laya model image.
The model image loads the pinned English base Laya checkpoint from HF's
`/repository` mount and serves decision scores through `/extract`. It also has an
Integrated Gradients path for candidate-logit attribution. Health capabilities
report implemented paths, not scientific qualification. Endpoint qualification
checks focus on whether the declared generation-time score, attribution target,
and token alignment are measured; they do not judge Laya's general correctness.

The endpoint package is independent of the experiments package's eager
training/plotting imports. It returns model signals only; the host-side
experiment runner computes drift through a pinned `expl_drift` revision.

## Local checks

From this directory, use Python 3.12 and the committed dependency lock:

```bash
rtk uv sync --locked
rtk uv run --locked pytest
rtk uv run --locked ruff check src tests scripts
rtk uv run --locked ruff format --check src tests scripts
rtk uv run --locked uvicorn expl_drift_endpoint.app:app --host 127.0.0.1 --port 8000
```

The tests exercise strict input validation, score ordering/serialization, IG
integration and token-span invariants, unsupported attribution methods, and
readiness with a test adapter. They use synthetic data and load no model weights.

## Container checks

Build the CPU-only validation image from this directory:

```bash
rtk proxy docker build --platform linux/amd64 --target scaffold -t expl-drift-endpoint:scaffold .
rtk proxy docker run --rm -p 127.0.0.1:8000:8000 expl-drift-endpoint:scaffold
```

The deployable `model-runtime` target installs the locked Laya SDK and CUDA
PyTorch dependencies, and starts `model_app`. It requires the pinned base
checkpoint mounted at `/repository` and a CUDA-capable HF endpoint. Do not use
the CPU scaffold as the model endpoint. Run the remote qualification checks on
the deployed GPU before interpreting outputs; `/health` capability names only
report implemented paths.

## CI checks and image publication

`.github/workflows/endpoint-ci.yml` runs on endpoint/workflow changes on
`codex/` pushes and pull requests. It checks the contract and builds/smoke-tests
the CPU-only `scaffold` target; it does not qualify real model inference.

`.github/workflows/publish-model-image.yml` is explicitly triggered with
`workflow_dispatch`. Run it from the desired source commit. It builds
the Linux amd64 `model-runtime` target and publishes
`docker.io/<Docker-Hub-username>/expl-drift-laya-endpoint:<full-commit-sha>`.
It reuses the Docker Hub credential names used by `scry`: `DOCKERHUB_USERNAME`
and `DOCKERHUB_TOKEN`. Those secrets must be available to this repository (or
through organization secrets), and the account must be able to push the image.
The job summary reports the immutable `@sha256:...` image reference. No HF
endpoint is created or changed by this workflow.

Use the immutable digest, not the tag, when configuring an HF endpoint. The
workflow has published the model-runtime image; endpoint image pull access was
verified on the research endpoint.

With the local CPU container running, execute the same smoke check from a second
terminal in this directory:

```bash
rtk proxy python3 scripts/smoke_container.py
```

The check verifies only CPU scaffold liveness and its intentional not-ready
responses. It does not exercise Laya or qualify the model-runtime image.

## Input contract

Schema version `1` accepts one instruction and ordered candidate set for a batch
of samples. Each request identifies its run/window and corpus SHA-256, and pins
model/tokenizer snapshots by full commit SHA. A separate head snapshot can be
supplied for CLM. Optional attributions require an explicit candidate target.

Input text is retained exactly. Duplicate sample/candidate IDs, unknown fields,
mutable revisions, unknown attribution targets, and implicit scalar conversions
are rejected. Current transport bounds are 32 samples, 64 candidates, and 8192
requested tokens; these are safety limits, not claims about model capacity.
The adapter enforces the configured checkpoint revision and the model's
`max_len`. Revision provenance comes from deployment configuration; mounted
checkpoint contents are not independently attested yet.

The current response schema returns each sample's predicted candidate ID and
ordered choice probabilities with `choice_probability` semantics. Requests may
include candidate-targeted Integrated Gradients with `candidate_logit`
semantics. IG requests default to 64. Adaptive composite Gauss-Legendre
integration estimates local error with 8- and 16-point rules, then bisects the
highest-error interval until the hard completeness gate passes or 16384 evaluations.
The response's `n_steps` field records total gradient evaluations and allows the full
16384-evaluation integration cap. This cap was raised after one sample failed
completeness at 8192 evaluations; the tolerance remains unchanged.
IG is rejected unless completeness residual is at most
`0.01 + 0.01 * abs(logit_delta)`. Token alignment, finite values, and truncation
counts are validated in the response. Health capability advertisement indicates
implementation availability only; it does not qualify IG.

## Current implementation status

The model extra pins `laya==0.3.29`; the lock fixes its Python dependencies.
The deployable image's model loader requires CUDA and does not fall back to CPU.
Model weights are not downloaded or executed locally; the pinned base model runs
on HF. The score path calls SDK `predict_batch`; IG uses the pinned SDK's private
sequence builder/collator and substitutes input embeddings while retaining the
decision head. It disables autocast so logits and gradients use the same
precision path. Adaptive composite Gauss-Legendre integration bisects the
highest-error interval until completeness passes or the 16384-evaluation cap is hit.
The remote pilot's 24 examples passed finite-value, token-alignment, truncation,
and hard completeness validation. Independent decision-score parity against a
native-SDK reference remains open. That comparison is needed to claim
equivalence to the SDK, not to observe the pinned endpoint's own outputs;
interpret those observations as endpoint behavior, not SDK-parity evidence.

### Remote pilot check on October 10 2026

- Image workflow: [38077186563](https://github.com/brendenmnoblitt/expl_drift_experiments/actions/runs/38077186563);
  digest `sha256:0ae8048118177879fa0cd228e4957214ca6459103bfbd9f00b9524db86501954`.
- HF `/health` returned 200 for the pinned base revision
  `7b928d828b7b0e022f929d9bd2e44165aa270148`.
- All 24 cached responses validated. Maximum actual IG cost was 2208 gradient
  evaluations; maximum absolute residual was 0.05124, with every sample inside
  its `0.01 + 0.01 * abs(logit_delta)` gate. Values were finite and token arrays
  aligned.
- Results and provenance: `../results/laya-youtube-spam-adaptive/`.

### Historical CPU scaffold check on October 7 2026

- Endpoint suite: 21 passed under Python 3.13.14; Ruff, format, and lock checks
  passed. Tests validate the numerical IG integration and the token-map
  contract, not real-model attribution.
- Local Uvicorn smoke passed for the CPU app; it confirms only expected scaffold
  behavior, not GPU model inference.
- Docker image build was not available because this WSL distro has no Docker
  Desktop integration. No model weights were downloaded, no GPU model was run,
  and no HF endpoint was deployed.

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
