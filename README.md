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
G1–G5. No primary results, frozen protocol, new deployment or paid extraction
are established by the development checks.

### Corrected corpus and immutable schedules

```bash
python -m experiments.research_extension.data --download \
  --output results/research-extension-data
```

The CPU-only command verifies the archive pin, strictly decodes UTF-8, preserves
exact source CONTENT, and writes inventories, source-ID/exact-text/template
links, exclusions, label/video counts, hashes, and provisional feasibility.
Source video is a retained grouping variable, not a video-disjoint holdout.
Role allocation requires all seven roles, including development, with explicit
counts and optional binary class counts. `schedule.allocate_roles` and
`schedule.build_schedule` allocate whole groups exactly or report infeasibility;
they never silently downsize, split incompatible roles, or repeat a group across
windows. Paired controls explicitly declare shared membership and transformations.

For saved role/window request JSON:

```bash
python -m experiments.research_extension.data --download \
  --output results/research-extension-data \
  --requests results/research-extension-data/role-requests.json \
  --windows results/research-extension-data/window-requests.json \
  --schedule-id youtube-candidate-design-seed42 --seed 42
```

Existing files are accepted only when identical; incompatible replacement fails.
The saved candidate design is not G2 freeze or null-budget qualification. Raw
inventories/schedules live under the ignored artifact directory, not in Git.

### Reference classifier development qualification

`ReferenceAdapter` loads only local BERT/GPT-2 task checkpoints and hashes their
files. It explains every fixed non-reference candidate margin, stores signed
document-only IG, complete forward quantities, logits/probabilities, token
metadata/truncation, last-layer document-mean embeddings, cost and runtime.
GPT-2's classifier token index is fixed from the original IDs throughout IG;
embedding-only default padding inference measures a different classifier.

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

## Remote Laya development and historical pilots

The active host runner uses [endpoint schema 2](endpoint/README.md): fixed
spam-minus-ham margins, unrounded temperature-scaled probabilities, document
representations, exact runtime/batch fingerprints, and independently
recalculable completeness. The data path now uses the corrected inventory and
one representative per duplicate/template group. Its legacy window-count
profiles remain development-only, not the seven-role primary study.

Configure a separately approved immutable schema-2 model-runtime image:

```bash
export LAYA_ENDPOINT_URL="https://<endpoint-url>"
export LAYA_ENDPOINT_IMAGE_DIGEST="sha256:<published-image-digest>"
export HF_TOKEN="<read token for the private endpoint>"
export LAYA_EXPERIMENT_OUTPUT="results/laya-youtube-spam-pilot-margin-v2-development"
python -m experiments.laya_endpoint.runner --profile pilot
```

Capabilities report implementation availability, not qualification. Changed
runtime, image, source, text, target or batch context cannot resume a run.
No deployment or remote extraction has been performed for schema 2.

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
