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

## Remote Laya Experiment

The host-side runner sends batches to a private Hugging Face Inference Endpoint; it does not load Laya locally. It pins the base model revision and UCI YouTube Spam Collection archive checksum, validates every response before caching, and resumes only when cached request fingerprints match.

With the endpoint configured to the immutable image digest that advertises both `decision_scores` and `integrated_gradients`, run:

Install the experiment dependencies from this repository's existing `requirements.txt` in the host Python environment; this installs analysis dependencies, not Laya or its model weights.

```bash
python -m pip install -r requirements.txt
export LAYA_ENDPOINT_URL="https://<endpoint-url>"
export LAYA_ENDPOINT_IMAGE_DIGEST="sha256:<published-image-digest>"
export HF_TOKEN="<read token for the private endpoint>"
python -m experiments.laya_endpoint.runner
```

The runner writes `results/laya-youtube-spam/provenance.json`, per-window validated responses, `experiment_results.json`, and `window_results.csv`. Its deterministic six-window class-prior shift is a small end-to-end check, not a direct comparison with the existing BERT/GPT AG News runs. Laya's complete training mix is undisclosed, so dataset non-overlap cannot be certified. Independent parity against Laya's native SDK also remains a qualification gate.

## Citation

- License: `LICENSE`
- Citation metadata: `CITATION.cff`

See sibling core library repo (`../expl_drift`) for detector/monitor implementation details.
