# expl_drift_experiments

Experiment and analysis package for the explanation-drift paper.

## Purpose

This repo contains:

- dataset loading and drift injection utilities
- model training/evaluation helpers
- notebooks for multi-seed statistical analysis
- generated figures/tables used in the manuscript

## Reproducing Main Results

Main analysis notebook:

- `notebooks/statistical_analysis.ipynb`
- one-command execution: `make paper-results`

Environment setup:

- install pinned dependencies: `pip install -r requirements.txt`

Current frozen run:

- `results/latest_run.txt` -> `stats_20260223_152707`
- reproducibility manifest:
  - `results/runs/stats_20260223_152707/repro_manifest.yaml`

Expected core configuration in the main notebook:

- `SEEDS = 25`
- `N_WINDOWS = 20`
- `DRIFT_START = 5`
- `N_CALIBRATION = 4`
- `WARNING_STD = 2.5`
- `CRITICAL_STD = 3.5`

Outputs for each run are saved under:

- `results/runs/stats_<timestamp>/figures`
- `results/runs/stats_<timestamp>/tables`

## Artifacts Used for Paper Drafting

For run `stats_20260223_152707`:

- aggregate metrics table:
  - `results/runs/stats_20260223_152707/tables/aggregate_summary.csv`
- per-seed/per-window raw results:
  - `results/runs/stats_20260223_152707/tables/full_results.csv`
- per-window mean/CI summary:
  - `results/runs/stats_20260223_152707/tables/per_window_stats.csv`
- all manuscript figures:
  - `results/runs/stats_20260223_152707/figures/*.png`

## Notes for Methods Section

- Synthetic experiments: Adult dataset with injected covariate/concept/mixed drift.
- Natural-drift experiment: Electricity dataset in chronological order.
- Class-conditional monitoring is compared against pooled monitoring.
- Lead time is measured as first significant accuracy drop minus first alert window.

Archived CUSUM/EWMA lead-time experiments are retained only as offline reference code and are not part of the active experiment runtime path.

## Seed Case Study Policy

- Primary claims are based on full multiseed aggregates, not individual-seed figures.
- Individual seed plots are supplemental illustrations selected from seeds with positive warning lead only.
- The exact selected seeds and figure paths are recorded in:
  - `results/runs/stats_20260223_152707/figures/seed_case_studies/case_study_seed_selection.csv`
- Do not replace case-study seeds manually after viewing figures; selection must remain tied to the recorded CSV.

## Research extension development

Active work is on `codex/research-extension`; the
[execution plan](../expl_drift/docs/research-extension-plan-2026-10-11.md) defines
G1–G5 remain open. One bounded schema-2 YouTube attempt on the prior runtime
completed 20 requests; request 21 timed out at 900 s and request 22 was not
sent. Cross-batch attribution exceeded tolerance, so no qualification report or
G1 pass exists. Logs and partial banks remain under
`results/research-extension-data/remote-laya-qualification-v2/`.

Commit `b9d69a1` changes scoring, representations, and fixed-margin IG to
singleton per-sample execution; endpoint tests pass (50). Its immutable image
`sha256:4c47ca3bb84b75f84c4a4a3bfbe8db79923205f58602ce34dbd6abe415077725`
was built by [workflow 38389756154](https://github.com/brendenmnoblitt/expl_drift_experiments/actions/runs/38389756154)
and configured on the existing L4 x1 endpoint without scaling changes. Its
`/health` returned schema 2 `ready` for the pinned Laya revision on NVIDIA L4.
The runtime reports `configured_not_attested`.
The user-authorized YouTube development comparison completed 22/22 requests
with 25,344 gradient evaluations and 17.9 minutes of runner time; the AG News
comparison completed 66/66 requests with 47,232 evaluations and 34.7 minutes.
Both compared eight designated development examples across batch sizes 1/4/8
and two repeats. Prediction parity passed, and maximum absolute differences
for logits, probabilities, representations, and attributions were all zero at
`atol=rtol=1e-5`. Reports:
`results/research-extension-data/remote-laya-qualification-v3/youtube-2026-10-13-singleton-b9d69a1/qualification.json`
and
`results/research-extension-data/remote-laya-qualification-v3/ag-news-2026-10-13-singleton-b9d69a1/qualification.json`.

The three historical observations use different checkpoint-artifact fingerprints
despite the same configured image and pinned model revision. Their reports did
not retain per-file manifests, so the exact historical changing path remains
unknown. Schema 3 matched loaded-weight hashes on two health-only starts but
omitted tokenizer and encoder-config identity.

Schema 4 added loaded weights, tokenizer settings, encoder config, and a full
non-hidden `/repository` manifest. Image
`sha256:b1c3babab9dca9481a87b30f67074a706dd50aee80a72a6f6cade20fd2184487`
from commit `e722e3ae6908c78606445fbd3b2432a0019828c0` was built by [workflow
38496718586](https://github.com/brendenmnoblitt/expl_drift_experiments/actions/runs/38496718586)
and deployed to the existing L4 x1 endpoint with scaling unchanged. Two
health-only cold starts matched at mount hash
`e76303c71f83ea241f7bc4fe844c903f5bbaf5f589c2a013fc4d9a4f8d9e9ee7`.

The subsequent bounded development qualification completed 22/22 YouTube and
66/66 AG News requests (72,576 gradient evaluations total, no failures). Both
tasks shared the exact schema-4 runtime fingerprint
`920827aef0e534d9c48fa5f99df5596e1f051fcf7ea68ed054813e2503cc5173`.
Batch sizes 1/4/8 and two repeats had prediction parity and zero maximum
absolute discrepancies for logits, probabilities, representations, and
attributions at `atol=rtol=1e-5`. Independently recomputed completeness gates
passed for 48/48 YouTube and 144/144 AG News attributions. The bank included
truncation in 1/8 YouTube and 4/8 AG News examples; slowest requests were
186.9s and 155.7s. Total runner wall time was 55.5 minutes; endpoint
resume-to-pause was 55.3 minutes, or about $0.74 at the listed $0.80/hour
before startup/minimums. User-approved target was $5; actual billing is
unavailable.

Per-file comparison of the health-only and extraction starts found the only
changed mount entry was `hub_model_info.json` (4,840 bytes; SHA-256
`bb19ecce8e7aa40497807a9630a26a4d359769d8f6aa490f24ee1c0d9b5e26bf` to
`c9152fed5b6e22ef4eed798aa32f8810d1953910b08d87eb8a15b9fc70a39cd2`);
the other 38 files matched. Hugging Face returned 404 for that filename at the
pinned model revision. Loaded weights, tokenizer, config, implementation, and
settings remained identical. This explains the observed schema-4 mount-hash
change and is consistent with, but does not prove, the source of the three
historical hashes. Schema 4 still includes mutable mount metadata in runtime
identity; distinct schema-4 fingerprints remain non-combinable.

Schema 5 separates semantic runtime identity (loaded model/tokenizer/config,
implementation, settings, software, device, and configured image) from the
diagnostic checkpoint-mount manifest returned by `/health`. Image
`sha256:f145d63fbf21a81eb954da37cd504c3d85308f411bfdb8dc62d85d529ddfcbb4`
was built from commit `3d0cb0861c884ae89ab6820d8fe05f570c3942b8` by [workflow
38564544055](https://github.com/brendenmnoblitt/expl_drift_experiments/actions/runs/38564544055)
and deployed to the existing L4 x1 endpoint with scaling unchanged. Two
health-only cold starts matched every semantic runtime field and the complete
39-file mount manifest, including `hub_model_info.json` (4,859 bytes,
SHA-256 `34830ef9c04d8080aa4245a359c601a3212c68c880e0b35c2902cb7785b6b804`).
The service reports `scientifically_qualified: false`.

Fresh signal qualification has not run. Both YouTube CLI invocations failed
local preflight before any HTTP request. The first lacked
`HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`; the single user-authorized renewed attempt
supplied the saved token but omitted `LAYA_ENDPOINT_URL` and
`LAYA_ENDPOINT_IMAGE_DIGEST`. Total extraction requests and gradients: zero.
The endpoint was paused after each failure. Further retry requires new explicit
authorization. Schema-4 results do not qualify schema 5; G1 remains open,
held-out evaluation remains sealed. Health evidence and both failures are in
`results/research-extension-data/remote-laya-qualification-v5/`. Full schema-4
evidence:
`results/research-extension-data/remote-laya-qualification-v4/attempt-summary.json`.
No main merge or paper tags changed.

### Corrected corpus and immutable schedules

```bash
python -m experiments.research_extension.data --download \
  --output results/research-extension-data/youtube-v2
```

The CPU-only command verifies the archive pin, strictly decodes UTF-8, preserves
exact source CONTENT, and writes inventories, source-ID/exact-text/template
links, exclusions, label/video counts, hashes, and provisional feasibility.
Source video is a retained grouping variable, not a video-disjoint holdout.
Role allocation requires all seven roles, including development, with explicit
counts and optional task-schema class counts. `schedule.allocate_roles` and
`schedule.build_schedule` allocate whole groups exactly or report infeasibility;
they never silently downsize, split incompatible roles, or repeat a group across
windows. Paired controls explicitly declare shared membership and transformations.

For saved role/window request JSON:

```bash
python -m experiments.research_extension.data --download \
  --output results/research-extension-data/youtube-v2 \
  --requests results/research-extension-data/role-requests.json \
  --windows results/research-extension-data/window-requests.json \
  --schedule-id youtube-candidate-design-seed42 --seed 42
```

Existing files are accepted only when identical; incompatible replacement fails.
The saved candidate design is not G2 freeze or null-budget qualification. Raw
inventories/schedules live under the ignored artifact directory, not in Git.

### Pinned AG News preparation

```bash
python -m experiments.research_extension.ag_news --download \
  --output results/research-extension-data/ag-news
```

The command verifies both Parquet files from `fancyzhx/ag_news` revision
`eb185aade064a813bc0b7f42de02595523103ca4` before decoding. PyArrow is an
explicit experiment dependency. Exact input text is retained; grouping uses
the same conservative case/whitespace/URL-template policy as YouTube.
Conflicting-label groups are excluded. Test records in any group containing
a training record are excluded, including template matches.

Pass `--requests /path/to/role-requests.json` to allocate all seven roles.
The allocator enforces original `train` for training and original `test` for
every other role, both at allocation and replay. Four-class counts use the
declared label schema; unavailable counts fail rather than changing the design.
Optional `--windows /path/to/window-requests.json --schedule-id <id>` emits
the same immutable schedule format as YouTube and requires `--requests`.
The CLI explicitly reports `protocol_frozen: false`.

Current preparation candidates, with equal class counts within each role:

| Role | AG News | YouTube |
| --- | ---: | ---: |
| Training | 3,000 | 300 |
| Development | 100 | 50 |
| Reference | 100 | 50 |
| Calibration | 400 | 200 |
| Threshold tuning | 1,000 | 200 |
| Held-out stable | 500 | 250 |
| Held-out evaluation | 1,000 | 500 |

These are not G2 allocations or adequate false-alarm evidence budgets.
AG News tuning supplies one disjoint ten-window, 100-example null horizon;
YouTube tuning supplies only four 50-example windows. Replays do not create
independent corpus observations. No held-out model outcomes are inspected.

### Pinned task-checkpoint training

Inventories now use schema 2 with explicit source partitions.
`Inventory.from_artifact` verifies the complete inventory hash and rejects
obsolete schemas; regenerate in a fresh directory rather than modifying old
candidate artifacts. Allocations are revalidated before loading weights.

```bash
python -m experiments.research_extension.training \
  --inventory /path/to/schema-2-inventory.json \
  --allocation /path/to/roles.json \
  --base-checkpoint /path/to/pinned-local-base \
  --base-sha256 <checkpoint-fingerprint> \
  --output /path/to/fresh-training-run \
  --epochs 3 --batch-size 8 --max-tokens 128 \
  --learning-rate 2e-5 --seed 42 --device cuda:0
```

Compute the base identity with
`experiments.research_extension.reference_adapter.checkpoint_fingerprint`.
Download public bases explicitly at an immutable revision before running;
the trainer is local-only. Prepared base identities and upstream revisions are
saved in the ignored `results/research-extension-data/base-checkpoints.json`.
The selected sources are `google-bert/bert-base-uncased` at
`86b5e0934494bd15c9632b12f734a8a67f723594` and `openai-community/gpt2` at
`607a30d783dfa663caf39e06633721c8d4cfcd7e`.

This is a new full-fine-tuning recipe, not a reproduction of historical
LoRA/Trainer checkpoints: float32, no autocast or TF32, eager attention,
deterministic algorithms, constant-rate AdamW and clipped gradients.
Only training updates weights. Minimum development log loss selects the
saved epoch; reference/tuning/held-out roles are never scored. BERT uses
`2e-5` and GPT-2 uses `5e-5` in the local three-epoch preparation run.
Both binary and four-class heads have their actual task labels; old four-class
weights are not merely renamed ham/spam.

Each fresh output saves the pre-inference training plan, selected
`checkpoint/` (model and tokenizer), and `training.json`: content identities,
role/group membership, software/source/settings, optimizer steps, development
accuracy/recall/log loss/Brier score, elapsed time and peak allocated GPU bytes.
Existing or partial output directories are refused, not silently resumed.
Training provenance alone is not G1 qualification.


### Reference classifier development qualification

`ReferenceAdapter` loads only local BERT/GPT-2 task checkpoints and hashes their
files. It explains every fixed non-reference candidate margin, stores signed
document-only IG, complete forward quantities, logits/probabilities, token
metadata/truncation, last-layer document-mean embeddings, cost and runtime.
GPT-2's classifier token index is fixed from the original IDs throughout IG;
embedding-only default padding inference measures a different classifier.
Completeness residuals use `math.fsum` over the actual persisted token values,
not a separate float32 tensor reduction. This preserves cancellation accuracy
and independently reproducible saved forward quantities. A changed extraction
source requires fresh banks; failed diagnostic banks remain untouched.

```bash
python -m experiments.research_extension.qualification \
  --checkpoint /path/to/task-checkpoint \
  --development-json /path/to/designated-development.json \
  --output /path/to/fresh-qualification-run --device cuda
```

The JSON supplies `role: "development"`, ordered `samples` with `sample_id`/`text`,
and `task` with ordered `candidate_ids`/`reference_candidate_id`. The command
saves tolerances before extraction, executes repeated batch sizes 1/4/8, and
saves exact banks plus a numerical/cost report. It refuses held-out roles and
already extracted destinations. Passing these examples alone does not pass G1.
Model extras and the separately installed endpoint numerical package are needed;
this is not the later installed-package/G5 cutover.

### Offline quality measures

`experiments.research_extension.quality.classification_quality` consumes saved
ordered raw logits, dense task labels, candidate IDs and the model's explicit
temperature. It reports accuracy, observed-class balanced accuracy, per-class
recall, fixed-prior standardized accuracy, log loss and multiclass Brier score.
Log loss is computed from float64 log-softmax, without clipping probabilities
that underflow to zero. No model or attribution calls occur.

Supply `reference_recalls` from the same fixed model to estimate the
composition-only accuracy expectation under the current empirical class prior.
This distinguishes prior composition from within-class changes; it does not
assume that a text shift causes degradation. Missing classes remain explicit:
observed-class balanced accuracy is labeled accordingly; standardized or
reference-composition expectations needing unavailable recalls are `None`,
with the affected class indices reported.

These are empirical point estimates, not group-aware intervals, underlying
degradation claims, quality-event null qualification, or G2 freeze. Label-only
controls reuse the exact same scores and alter host-side labels only.


## Remote Laya development and historical pilots

The active host runner targets endpoint schema 5: fixed
spam-minus-ham margins, unrounded temperature-scaled probabilities, document
representations, semantic runtime/batch fingerprints, and independently
recalculable completeness. The full checkpoint-mount manifest is a health
diagnostic, not a cache-key component. The schema-5 image is deployed on the
existing endpoint; two health-only cold starts matched semantic identity and
all 39 checkpoint files.
Two YouTube attempts failed local preflight before any HTTP request: first the
token variable was absent; the renewed wrapper later omitted URL and image
digest variables. The endpoint is paused; no extraction or window study ran,
and another attempt requires renewed authorization.
The data path uses the corrected inventory and one representative per
duplicate/template group. Legacy window-count profiles remain development-only,
not the seven-role primary study.
The development qualifier is endpoint-only; it does not load a local model.
It requires exactly eight designated development examples for the selected
corpus and preserves its frozen candidate order/reference. YouTube runs the
spam-minus-ham margin; AG News runs each of the three non-reference class
margins against world. It requires schema 5 from `/health`, the pinned model,
required capabilities, and the requested immutable image digest before
extraction. Run only after explicit approval for the corresponding request and
gradient-evaluation ceiling:

```bash
export LAYA_ENDPOINT_URL="https://<endpoint-url>"
export LAYA_ENDPOINT_IMAGE_DIGEST="sha256:<published-image-digest>"
export HF_TOKEN="<read token for the private endpoint>"
python -m experiments.research_extension.laya_qualification \
  --task youtube \
  --development-json results/research-extension-data/youtube-v2/bounded-development-qualification.json \
  --output results/research-extension-data/remote-laya-qualification-v3/youtube-<unique-run-id> \
  --allow-remote-execution
```

AG News uses a separate run:

```bash
python -m experiments.research_extension.laya_qualification \
  --task ag_news \
  --development-json results/research-extension-data/ag-news/bounded-development-qualification.json \
  --output results/research-extension-data/remote-laya-qualification-v3/ag-news-<unique-run-id> \
  --allow-remote-execution
```

Each task repeats batch sizes 1, 4 and 8 twice, sets a 15-minute per-request
timeout and a 2-hour total wall-time ceiling, and never retries. The YouTube
binary task makes at most 22 extraction requests (8 examples × 6 passes), with
a 786,432-gradient-evaluation ceiling. The four-candidate AG News task makes
at most 66 requests, with a 2,359,296-evaluation ceiling. Requested IG uses
64 steps; the service caps adaptive work at 16,384 evaluations per target and
example. Each successful response and outcome is fsynced before the next
request. A failed run is not resumed; start a separately identified fresh run
only after inspecting its saved records. These development comparisons do not
pass G1.

Preflight of the supplied HF endpoint on 2026-10-12 found it paused. After
explicit restart approval, `/health` returned schema 1 for the pinned Laya
revision, with only `decision_scores` and `integrated_gradients`; it supplied
no image digest and did not implement the schema-2 fixed-margin and document
representation contract. That historical preflight is preserved at
`results/research-extension-data/remote-laya-qualification-v1/youtube-preflight-2026-10-12.json`.

With subsequent deployment approval, the existing `laya-egk` endpoint was
updated to the schema-2 image built from commit
`e9850308794b938c2423e5d0f8eb1241b0cf73a1` (GitHub Actions run
`38331632302`, digest
`sha256:80419fbb20ea42e6c8c1cafb29988c329c7c3b221dc6c6bb142abe8cfa2f49b5`).
Its hardware and scaling configuration were preserved. Readiness verified the
pinned Laya revision and schema-2 capabilities. The runtime reports image
identity as configured, not independently attested.

The subsequent YouTube development run is incomplete:
`results/research-extension-data/remote-laya-qualification-v2/youtube-2026-10-13/`.
Twenty requests completed and their responses were persisted; request 21 timed
out after 900 seconds, and request 22 was not sent. Repeats were exactly
identical within batch sizes 1 and 4. The cross-batch comparison failed:
predictions, logits, probabilities and document representations met the
comparison tolerances, but attribution values differed by up to 0.046.
All 32 available completeness checks passed; completed responses reported
16,992 gradient evaluations. No retry or resume occurred; no full comparison
report or G1 qualification was produced. The endpoint was paused; preserve the
partial outputs and never resume them under the changed runtime. Fresh resource
approval was received for the separate run below.

The endpoint now executes scoring, representation, and fixed-margin IG in a
singleton context per sample. The new image was built from `b9d69a1`,
configured on the existing L4 x1 endpoint, health-checked, then paused.
The singleton-runtime YouTube run
`results/research-extension-data/remote-laya-qualification-v3/youtube-2026-10-13-singleton-b9d69a1/`
completed 22/22 requests with no retries, 25,344 gradient evaluations, and
17.9 minutes of runner wall time. The AG News run
`results/research-extension-data/remote-laya-qualification-v3/ag-news-2026-10-13-singleton-b9d69a1/`
completed 66/66 requests, 47,232 gradient evaluations, and 34.7 minutes of
runner wall time. Both compared eight designated development examples at
batch sizes 1/4/8 with two repeats; predictions matched and all maximum
absolute signal discrepancies were zero at `atol=rtol=1e-5`. See each
`qualification.json`, its request/response/event artifacts, and
`attempt-summary.json`.

The endpoint returned different checkpoint-artifact fingerprints across
restarts (`35b599d63dfa3816f29a8c1280499560e4950b5a5ebf6b75d1437a2769e1abd1`
for YouTube,
`f81510fe53d225e1553f302b08fc724752b1c535cc6c0325320dd33dcb7f3c1a` for AG
News). A health-only cold-start check at
`results/research-extension-data/remote-laya-qualification-v3/runtime-fingerprint-check-2026-10-13/`
returned a third hash,
`a3f80ff3984e86b0d48879cbfed9c8bd4025d5f0c8e8729227a1d7f653d23b01`;
zero extraction requests were made, and the endpoint was paused. Runtime
identity hashes every non-hidden file under `/repository` after model
initialization; the source of the variance remains unknown. Keep the task runs
separate; neither establishes same-runtime cross-task qualification or G1. No
held-out evaluation ran.

The interrupted local native-SDK artifacts under
`results/research-extension-data/native-laya-qualification-v2/` are preserved
as historical diagnostics. They are not resumed, merged with endpoint output,
or accepted as endpoint qualification.

A separate exploratory pilot runner is not the bounded qualifier above; it
requires its own explicit request/cost approval and a fresh output path:

```bash
export LAYA_ENDPOINT_URL="https://<endpoint-url>"
export LAYA_ENDPOINT_IMAGE_DIGEST="sha256:<published-image-digest>"
export HF_TOKEN="<read token for the private endpoint>"
export LAYA_EXPERIMENT_OUTPUT="results/laya-youtube-spam-pilot-margin-v2-development"
python -m experiments.laya_endpoint.runner --profile pilot
```

Capabilities report implementation availability, not qualification. Changed
runtime, image, source, text, target or batch context cannot resume a run.
Schema-2 endpoint deployment and extraction have since been performed as
recorded above; the capability list alone still does not establish qualification.

The following results are historical schema-1 candidate-logit pilots using the
old text/sampling/metric definitions, not corrected margin evidence.
Their original files remain in `results/laya-youtube-spam-adaptive/` and
`results/laya-youtube-spam-calibrated/`; the new runner rejects those caches.

The historical 24-example smoke used a deterministic class-prior shift,
not a direct BERT/GPT comparison. The detector returned no alert in the shifted
windows:

| Window | Ham / spam | Accuracy | Alert |
|---:|---:|---:|---|
| 0 | 2 / 2 | 1.00 | calibration |
| 1 | 2 / 2 | 1.00 | calibration |
| 2 | 2 / 2 | 0.50 | calibration |
| 3 | 1 / 3 | 0.75 | ok |
| 4 | 0 / 4 | 0.75 | ok |
| 5 | 0 / 4 | 0.50 | ok |

Baseline accuracy was 1.00; mean accuracy over later windows was 0.70. Four
examples per window are too few for an inference about detector behavior. UCI
YouTube Spam is not listed among the pinned model-card benchmarks, but Laya's
full training mix is undisclosed, so overlap cannot be certified. Native-SDK
score parity is still unverified; that limits claims that this adapter matches
the SDK, not observations of outputs from the pinned endpoint itself.

### Threshold replay and larger follow-up

Replaying the first run's validated cache through `DriftMonitor` reconstructs
these calibration thresholds. The original result artifact did not save them:

| Metric | Calibration mean | Calibration std | Warning threshold | Largest shifted-window value |
|---|---:|---:|---:|---:|
| `cosine_drift` | 0.000822 | 0.000809 | 0.002845 | 0.000495 |
| `max_jsd` | 0.606504 | 0.086643 | 0.823112 | 0.693147 |
| `max_wasserstein` | 13.500000 | 10.000000 | 38.500000 | 11.000000 |

With natural-log entropy, JSD is bounded by `ln(2) ≈ 0.693147`. The pilot's
JSD warning threshold is above that ceiling, so that metric could not fire in
this run. Calibration used only two four-example windows.

The historical `calibrated` profile used 100 samples in each of ten windows:
one baseline, four calibration windows, then spam prevalence rising from 60%
to 100%. Row disjointness did not establish duplicate/template disjointness.
The partial cache mixed images and used Latin-1-transformed inputs; preserve it
as historical, never resume it as corrected evidence.

During this run, HF's scale-to-zero timeout was temporarily raised from 15 to
60 minutes after the endpoint scaled down during a long extraction request.
Minimum replicas remain zero; restore the timeout to 15 minutes after the run.

For a new development-only schema-2 run, use a fresh destination:

```bash
LAYA_EXPERIMENT_OUTPUT="results/laya-youtube-spam-calibrated-margin-v2-development" \
  python -m experiments.laya_endpoint.runner --profile calibrated
```

The `calibrated` run remains exploratory. Partial results show explanation
distribution changes and detector alerts on the deployed endpoint's outputs
under the synthetic class-prior shift. The run also has task labels and
per-window accuracy measurements, but it does not establish that alerts
reliably track production degradation: that requires a systematic comparison
of alert behavior with an independent quality or health measure over matching
windows. Do not generalize this synthetic input-mix result to production
performance.

## Citation

- License: `LICENSE`
- Citation metadata: `CITATION.cff`

See sibling core library repo (`../expl_drift`) for detector/monitor implementation details.
