from __future__ import annotations

from importlib import import_module

import pytest

runner = import_module("experiments.laya_endpoint.runner")


def test_calibrated_profile_has_four_stable_windows_and_gradual_prior_shift():
    counts = runner.CALIBRATED_WINDOW_CLASS_COUNTS

    assert len(counts) == 10
    assert counts[:5] == [{0: 50, 1: 50}] * 5
    assert [window[1] / 100 for window in counts[5:]] == [0.6, 0.7, 0.8, 0.9, 1.0]
    assert runner.EXPERIMENT_PROFILES["calibrated"]["calibration_windows"] == 4


def test_calibrated_windows_use_distinct_rows_and_exact_class_counts():
    records = [
        {"sample_id": f"ham-{index}", "text": f"ham {index}", "label": 0} for index in range(400)
    ] + [
        {"sample_id": f"spam-{index}", "text": f"spam {index}", "label": 1} for index in range(700)
    ]

    windows = runner._create_windows(records, runner.CALIBRATED_WINDOW_CLASS_COUNTS)

    assert [len(window) for window in windows] == [100] * 10
    assert [
        (sum(row["label"] == 0 for row in window), sum(row["label"] == 1 for row in window))
        for window in windows
    ] == [(counts[0], counts[1]) for counts in runner.CALIBRATED_WINDOW_CLASS_COUNTS]
    sample_ids = [row["sample_id"] for window in windows for row in window]
    assert len(sample_ids) == len(set(sample_ids))


def test_endpoint_chunks_stay_within_contract_limit_and_cover_samples_once():
    samples = [{"sample_id": str(index), "text": f"comment {index}"} for index in range(100)]

    chunks = runner._window_chunks(samples)

    assert [len(chunk) for chunk in chunks] == [32, 32, 32, 4]
    assert [sample for chunk in chunks for sample in chunk] == samples


def test_resume_provenance_tracks_image_and_runner_revisions():
    old_image = "sha256:" + "a" * 64
    new_image = "sha256:" + "b" * 64
    previous = {
        "dataset": {"archive_sha256": "d" * 64},
        "endpoint": {"url": "https://endpoint", "image_digest": old_image},
        "experiment": {
            "profile": "calibrated",
            "endpoint_request_batch_sizes": [32],
        },
        "revisions": {
            "expl_drift": "e" * 40,
            "expl_drift_experiments": "f" * 40,
        },
    }
    current = {
        "dataset": {"archive_sha256": "d" * 64},
        "endpoint": {"url": "https://endpoint", "image_digest": new_image},
        "experiment": {
            "profile": "calibrated",
            "endpoint_request_batch_sizes": [8],
        },
        "revisions": {
            "expl_drift": "e" * 40,
            "expl_drift_experiments": "1" * 40,
            "experiment_code_revisions": ["1" * 40],
        },
    }

    merged, legacy_image = runner._merge_resume_provenance(
        previous, current, resume_from_image_digest=old_image
    )

    assert merged["endpoint"]["image_digests"] == [old_image, new_image]
    assert merged["endpoint"]["legacy_cache_image_digest"] == old_image
    assert legacy_image == old_image
    assert merged["revisions"]["experiment_code_revisions"] == ["f" * 40, "1" * 40]
    assert merged["experiment"]["endpoint_request_batch_sizes"] == [32, 8]


def test_resume_provenance_requires_explicit_previous_image():
    old_image = "sha256:" + "a" * 64
    current_image = "sha256:" + "b" * 64
    previous = {
        "dataset": {},
        "endpoint": {"url": "https://endpoint", "image_digest": old_image},
        "experiment": {},
        "revisions": {"expl_drift": "e" * 40, "expl_drift_experiments": "f" * 40},
    }
    current = {
        "dataset": {},
        "endpoint": {"url": "https://endpoint", "image_digest": current_image},
        "experiment": {},
        "revisions": {
            "expl_drift": "e" * 40,
            "expl_drift_experiments": "1" * 40,
            "experiment_code_revisions": ["1" * 40],
        },
    }

    with pytest.raises(ValueError, match="LAYA_RESUME_PREVIOUS_IMAGE_DIGEST"):
        runner._merge_resume_provenance(previous, current, resume_from_image_digest=None)
