"""Legacy development scheduling remains separate from primary group allocation."""

import copy

import pytest
from experiments.laya_endpoint import runner


def test_development_sampling_has_exact_class_counts_without_row_reuse():
    records = [
        {
            "sample_id": f"{label}-{i}-{variant}",
            "group_id": f"{label}-{i}",
            "text": f"text {label} {i} {variant}",
            "label": label,
        }
        for label in (0, 1)
        for i in range(8)
        for variant in range(2)
    ]
    windows = runner._create_windows(records, [{0: 2, 1: 3}, {0: 3, 1: 2}])
    assert [sum(row["label"] == 1 for row in window) for window in windows] == [3, 2]
    ids = [row["sample_id"] for window in windows for row in window]
    assert len(ids) == len(set(ids))
    groups = [row["group_id"] for window in windows for row in window]
    assert len(groups) == len(set(groups))
    with pytest.raises(ValueError):
        runner._create_windows(records, [{0: 9, 1: 0}])


@pytest.mark.parametrize("change", ["image", "source", "batch", "text"])
def test_new_artifacts_never_resume_under_changed_provenance(change):
    prior = {
        "dataset": {"text_sha256": "a" * 64},
        "endpoint": {"image_digest": "sha256:" + "b" * 64},
        "revisions": {"analysis_source": "c" * 40},
        "experiment": {"batch_size": 8},
    }
    current = copy.deepcopy(prior)
    if change == "image":
        current["endpoint"]["image_digest"] = "sha256:" + "d" * 64
    elif change == "source":
        current["revisions"]["analysis_source"] = "d" * 40
    elif change == "batch":
        current["experiment"]["batch_size"] = 4
    else:
        current["dataset"]["text_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        runner._merge_resume_provenance(prior, current)
