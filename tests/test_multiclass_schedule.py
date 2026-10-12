"""Task-schema labels and corpus-split constraints in the exact group allocator."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from experiments.research_extension.data import Inventory, SourceRecord, sha256_json
from experiments.research_extension.schedule import (
    REQUIRED_ROLES,
    AllocationInfeasible,
    _partition,
    allocate_roles,
    build_schedule,
    provisional_feasibility,
)


def inventory_fixture(*, labels=(0, 1, 2, 3), present=None, policy=True, group_size=1):
    records = []
    present = labels if present is None else present
    for split, copies in (("train", 4), ("test", 16)):
        for label in present:
            for index in range(copies):
                group = f"{split}-{label}-{index}"
                for row in range(group_size):
                    source = f"{group}-{row}"
                    records.append(
                        SourceRecord(
                            source_id=source,
                            comment_id=source,
                            video="",
                            member=f"data/{split}.parquet",
                            row_number=len(records) + 1,
                            text=f"Original text {source}\n",
                            label=label,
                            source_label=str(label),
                            group_id=group,
                            exclusions=(),
                            source_split=split,
                        )
                    )
    schema = {
        "version": 1,
        "candidate_ids": [f"class-{label}" for label in labels],
        "labels": {str(label): f"class-{label}" for label in labels},
        "reference_candidate": f"class-{labels[0]}",
    }
    manifest = {
        "schema_version": 2,
        "archive_sha256": sha256_json("fixture archive"),
        "task_schema": schema,
        "transforms": {"input": "identity"},
        "hashes": {
            "task_schema": sha256_json(schema),
            "transforms": sha256_json({"input": "identity"}),
        },
    }
    if policy:
        manifest["role_source_splits"] = {
            role: "train" if role == "training" else "test" for role in REQUIRED_ROLES
        }
    groups = tuple({"group_id": group} for group in sorted({row.group_id for row in records}))
    manifest["inventory_sha256"] = sha256_json(
        {"manifest": manifest, "records": [asdict(row) for row in records], "groups": groups}
    )
    return Inventory(tuple(records), groups, manifest)


def balanced_requests(labels=(0, 1, 2, 3), each=1):
    return {
        role: {"count": len(labels) * each, "class_counts": {label: each for label in labels}}
        for role in REQUIRED_ROLES
    }


def reseal_allocation(allocation):
    allocation["membership_sha256"] = sha256_json(
        {role: data["source_ids"] for role, data in allocation["roles"].items()}
    )
    allocation["allocation_sha256"] = sha256_json(
        {key: value for key, value in allocation.items() if key != "allocation_sha256"}
    )


def test_four_class_balanced_roles_enforce_disjoint_source_splits():
    inventory = inventory_fixture()
    allocation = allocate_roles(inventory, balanced_requests(), seed=9)
    assert allocation == allocate_roles(inventory, balanced_requests(), seed=9)
    assert allocation["role_source_splits"] == inventory.manifest["role_source_splits"]
    by_source = {row.source_id: row for row in inventory.records}
    all_groups = []
    for role, data in allocation["roles"].items():
        assert data["count"] == 4
        assert data["class_counts"] == {"0": 1, "1": 1, "2": 1, "3": 1}
        assert data["videos"] == {}
        assert {by_source[source].source_split for source in data["source_ids"]} == {
            "train" if role == "training" else "test"
        }
        all_groups.extend(data["group_ids"])
    assert len(all_groups) == len(set(all_groups))


def test_absent_schema_class_normalizes_to_zero_and_positive_request_refuses():
    inventory = inventory_fixture(present=(0, 1, 2))
    requests = balanced_requests((0, 1, 2))
    allocation = allocate_roles(inventory, requests)
    assert allocation["requests"]["training"]["class_counts"] == {"0": 1, "1": 1, "2": 1, "3": 0}
    with pytest.raises(AllocationInfeasible) as error:
        allocate_roles(inventory, balanced_requests())
    assert error.value.report["available"]["class_counts"]["3"] == 0
    schedule = build_schedule(
        inventory,
        allocation,
        [{"role": "reference", "count": 3, "class_counts": {0: 1, 1: 1, 2: 1}}],
        seed=1,
        schedule_id="absent-class",
    )
    assert schedule["window_requests"][0]["class_counts"]["3"] == 0


@pytest.mark.parametrize("label", [True, False, -1, "-1", 4, "4", 1.0, "01"])
def test_invalid_or_unknown_request_label_refuses(label):
    requests = balanced_requests()
    requests["training"] = {"count": 1, "class_counts": {label: 1}}
    with pytest.raises(ValueError):
        allocate_roles(inventory_fixture(), requests)


def test_duplicate_integer_and_string_class_request_refuses():
    requests = balanced_requests()
    requests["training"] = {"count": 2, "class_counts": {0: 1, "0": 1}}
    with pytest.raises(ValueError, match="duplicate"):
        allocate_roles(inventory_fixture(), requests)


@pytest.mark.parametrize("count", [True, -1, 1.0])
def test_invalid_class_count_refuses(count):
    requests = balanced_requests()
    requests["training"]["class_counts"][0] = count
    with pytest.raises(ValueError):
        allocate_roles(inventory_fixture(), requests)


def test_training_cannot_borrow_available_test_examples():
    requests = balanced_requests()
    requests["training"] = {"count": 20, "class_counts": {label: 5 for label in range(4)}}
    with pytest.raises(AllocationInfeasible) as error:
        allocate_roles(inventory_fixture(), requests)
    assert error.value.report["available"]["examples"] == 16
    assert error.value.report["requested"]["training"]["count"] == 20


def test_impossible_whole_group_counts_refuse_without_downsize():
    with pytest.raises(AllocationInfeasible, match="no exact simultaneous") as error:
        allocate_roles(inventory_fixture(group_size=2), balanced_requests())
    assert error.value.report["group_size_gcd"] == 2


@pytest.mark.parametrize("split", ["", "collection", "validation", "train"])
def test_unsupported_missing_or_mixed_group_source_split_refuses(split):
    inventory = inventory_fixture(group_size=2)
    records = list(inventory.records)
    test_index = next(index for index, row in enumerate(records) if row.source_split == "test")
    records[test_index] = replace(records[test_index], source_split=split)
    inventory = replace(inventory, records=tuple(records))
    with pytest.raises(ValueError, match="source split membership"):
        allocate_roles(inventory, balanced_requests(each=2))


def test_incomplete_role_split_policy_refuses():
    inventory = inventory_fixture()
    del inventory.manifest["role_source_splits"]["development"]
    with pytest.raises(ValueError, match="every mandatory role"):
        allocate_roles(inventory, balanced_requests())


def test_rehashed_train_test_role_leakage_refuses_on_replay():
    inventory = inventory_fixture()
    allocation = allocate_roles(inventory, balanced_requests())
    allocation["roles"]["training"], allocation["roles"]["reference"] = (
        allocation["roles"]["reference"],
        allocation["roles"]["training"],
    )
    reseal_allocation(allocation)
    with pytest.raises(ValueError, match="source split policy violation"):
        build_schedule(inventory, allocation, [], seed=1, schedule_id="leaked")


@pytest.mark.parametrize("policy", [None, {}, {role: "test" for role in REQUIRED_ROLES}])
def test_rehashed_removed_or_changed_allocation_policy_refuses(policy):
    inventory = inventory_fixture()
    allocation = allocate_roles(inventory, balanced_requests())
    if policy is None:
        del allocation["role_source_splits"]
    else:
        allocation["role_source_splits"] = policy
    reseal_allocation(allocation)
    with pytest.raises(ValueError, match="source split policy"):
        build_schedule(inventory, allocation, [], seed=1, schedule_id="wrong-policy")


def test_four_class_windows_remain_disjoint_and_preserve_paired_controls():
    inventory = inventory_fixture()
    requests = balanced_requests()
    requests["held_out_evaluation"] = balanced_requests(each=2)["held_out_evaluation"]
    allocation = allocate_roles(inventory, requests)
    windows = [
        {
            "role": "held_out_evaluation",
            "count": 4,
            "class_counts": {label: 1 for label in range(4)},
        }
        for _ in range(2)
    ]
    schedule = build_schedule(inventory, allocation, windows, seed=3, schedule_id="four-class")
    groups = [group for window in schedule["windows"] for group in window["group_ids"]]
    assert len(groups) == len(set(groups)) == 8
    assert all(
        window["class_counts"] == {str(label): 1 for label in range(4)}
        for window in schedule["windows"]
    )
    variants = {
        sample["source_id"]: {"label": (sample["label"] + 1) % 4}
        for window in schedule["windows"]
        for sample in window["samples"]
    }
    paired = build_schedule(
        inventory,
        allocation,
        windows,
        seed=11,
        schedule_id="four-class-control",
        paired_with=schedule,
        variants=variants,
        reuse_declaration={"kind": "same source windows"},
        transformation={"labels": "cyclic four-class permutation"},
    )
    assert paired["membership_sha256"] == schedule["membership_sha256"]
    assert paired["text_sha256"] == schedule["text_sha256"]
    assert paired["labels_sha256"] != schedule["labels_sha256"]
    with pytest.raises(AllocationInfeasible):
        build_schedule(
            inventory, allocation, windows + [windows[0]], seed=3, schedule_id="oversubscribed"
        )
    bad_window = deepcopy(windows[0])
    bad_window["class_counts"] = {4: 4}
    with pytest.raises(ValueError):
        build_schedule(inventory, allocation, [bad_window], seed=3, schedule_id="unknown-class")


def test_noncontiguous_nonnegative_label_identities():
    labels = (0, 2, 5, 9)
    inventory = inventory_fixture(labels=labels)
    allocation = allocate_roles(inventory, balanced_requests(labels))
    schedule = build_schedule(
        inventory,
        allocation,
        [{"role": "reference", **balanced_requests(labels)["reference"]}],
        seed=1,
        schedule_id="noncontiguous",
    )
    assert schedule["windows"][0]["class_counts"] == {str(label): 1 for label in labels}


def test_binary_whole_corpus_matches_same_exact_engine_and_video_counts():
    inventory = inventory_fixture(labels=(0, 1), policy=False)
    inventory = replace(
        inventory, records=tuple(replace(row, video="video") for row in inventory.records)
    )
    requests = balanced_requests((0, 1))
    groups = {}
    for row in inventory.eligible_records:
        groups.setdefault(row.group_id, []).append(row)
    expected = _partition(groups, requests, 42)
    allocation = allocate_roles(inventory, requests)
    assert "role_source_splits" not in allocation
    assert {role: data["group_ids"] for role, data in allocation["roles"].items()} == expected
    assert all(data["videos"] == {"video": 2} for data in allocation["roles"].values())
    windows = [{"role": "reference", "count": 2, "class_counts": {0: 1, 1: 1}}]
    schedule = build_schedule(inventory, allocation, windows, seed=1, schedule_id="binary")
    assert Counter(sample["label"] for sample in schedule["windows"][0]["samples"]) == {0: 1, 1: 1}


def test_provisional_feasibility_is_not_an_ag_news_allocator():
    with pytest.raises(ValueError, match="binary/YouTube-only"):
        provisional_feasibility(inventory_fixture())


def test_multiclass_exact_search_backtracks_instead_of_greedily_failing():
    groups = {
        name: [replace(inventory_fixture().records[0], label=5)] * size
        for name, size in (("three", 3), ("two-a", 2), ("two-b", 2), ("two-c", 2))
    }
    assigned = _partition(
        groups,
        {"large": {"count": 6, "class_counts": {5: 6}}, "small": {"count": 3}},
        7,
        (0, 5),
    )
    assert assigned == {"large": ["two-a", "two-b", "two-c"], "small": ["three"]}
