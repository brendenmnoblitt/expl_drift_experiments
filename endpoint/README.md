# Laya extraction endpoint

The endpoint has a CPU-only validation image and a separate Laya model image.
The model image loads the pinned English base Laya checkpoint from HF's
`/repository` mount and serves decision scores through `/extract`. It also has an
Integrated Gradients path for fixed candidate-minus-reference logit margins.
Health capabilities report implemented paths, not scientific qualification.
Endpoint qualification
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

The tests exercise strict input validation, fixed signed margins, saved-forward
completeness, document-only pooling, exact cache/runtime identity, HTTP errors,
and adaptive integration. They use CPU numerical fixtures, not task checkpoints.

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

Use the immutable digest, not the tag, when configuring an HF endpoint. Set
`model.env.LAYA_ENDPOINT_IMAGE_DIGEST` to the same `sha256:...` value in the
endpoint update; the model service fails startup if this required value is
missing. This configured identity is not independently attested.

With the local CPU container running, execute the same smoke check from a second
terminal in this directory:

```bash
rtk proxy python3 scripts/smoke_container.py
```

The check verifies only CPU scaffold liveness and its intentional not-ready
responses. It does not exercise Laya or qualify the model-runtime image.

## Input contract

Schema version `2` accepts one instruction and ordered candidate set per request.
Every sample is collated and executed independently in a fixed singleton context,
regardless of its transport batch size. It pins model/tokenizer snapshots, the
observed `/health` runtime, `precision="float32_no_autocast"`,
`tokenization="laya_sdk_0.3.29_encode_state"`, and
`pooling="document_mean_last_hidden_state"`. `decision_target` specifies
`target_semantics="fixed_candidate_margin"`, `target_candidate_id`, and a
distinct `reference_candidate_id`. The target never follows the prediction.

Input text is retained exactly. Duplicate IDs, unknown fields, mutable revisions,
unknown targets, and scalar coercions are rejected. Transport bounds remain
32 samples, 64 candidates, and 8192 requested tokens; the adapter also enforces
the loaded model's `max_len`.

Every sample returns ordered raw logits, unrounded probabilities computed from
the same per-sample forward logits and its saved SDK probability temperature, a
fixed margin value, and aligned token/segment/truncation metadata. Requested
representations average that singleton forward's encoder state over document
tokens only; no retained document tokens means an error, not a fabricated
embedding. Integrated gradients use the same singleton context, embedding path,
raw decision head and float32 arithmetic, without re-evaluating unrelated request
samples at each integration step. It saves signed values, input/baseline target
values, requested steps, actual gradient evaluations, and the residual:

`input_target_value - baseline_target_value - sum(document_attributions)`.

Adaptive 8/16-point Gauss-Legendre panels refine until the implementation gate
`abs(residual) <= 0.01 + 0.01 * abs(input_target_value - baseline_target_value)`
passes, or the 16384-evaluation cap fails. This is a numerical check, not a
faithfulness or scientific qualification claim.

The response fingerprints the entire canonical request and ordered batch.
Runtime identity hashes mounted checkpoint artifacts, service source and loaded
settings, and records SDK/Python/Torch/Transformers/device identities. Set
`LAYA_ENDPOINT_IMAGE_DIGEST` in the service environment; it is explicitly
`configured_not_attested`, not a cryptographically verified running image.
The host independently recalculates probabilities, margins and completeness.
Split requests retain each child's exact execution identity; concatenation does
not claim that the parent batch was executed. Cache/resume rejects changed text,
targets, precision, runtime, image, and batch context. Schema-1 candidate-logit
and mixed-image pilot caches are preserved historical artifacts, never resumed
as schema-2 margins.

## Current implementation status

The model extra pins `laya==0.3.29`; the lock fixes its Python dependencies.
The deployable image's model loader requires CUDA and does not fall back to CPU.
The adapter calls the pinned SDK's actual decision model after its sequence
builder/collator, rather than mixing rounded `predict_batch` probabilities with
singleton logit IG. Scoring, IG forward endpoints, and document pooling now share
one declared numerical path. Capabilities describe implemented signals only:
`scientifically_qualified` remains false. Real task-checkpoint, batching,
alignment, pooling and signed/permutation G1 checks are still required on one
immutable runtime before primary extraction.

The schema-1 pilots below predate this implementation. Their finite-value,
alignment and candidate-logit completeness evidence does not qualify schema-2
margin or embedding signals. The current image deployment is recorded in the
root README; these historical results do not imply it.

### Incomplete development attempt and fix

- The October 13 YouTube development attempt used image
  `sha256:80419fbb20ea42e6c8c1cafb29988c329c7c3b221dc6c6bb142abe8cfa2f49b5`
  from workflow [38331632302](https://github.com/brendenmnoblitt/expl_drift_experiments/actions/runs/38331632302).
  Twenty requests completed; request 21 timed out after 900 seconds, and request
  22 was not sent. No report or G1 pass was produced.
- Repeated requests at each of batch sizes 1 and 4 matched exactly. Cross-batch
  attributions differed by up to `0.0459985` against `1e-5` tolerances, despite
  passing completeness checks. The batch-size-8 comparison is missing. Preserve
  the logs and partial banks; do not resume them under another runtime.
- Commit `b9d69a1` removes transport-batch-dependent model context: scoring,
  representations, and fixed-margin IG execute per sample in a singleton
  context. The endpoint suite passes (50 tests). A new immutable image is
  building under workflow
  [38389756154](https://github.com/brendenmnoblitt/expl_drift_experiments/actions/runs/38389756154).
  Fresh qualification is still required; no additional extraction is authorized
  by this fix.

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
