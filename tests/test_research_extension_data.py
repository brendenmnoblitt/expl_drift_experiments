"""Semantic boundaries for corrected source inventory and whole-group schedules."""

import csv
import hashlib
import io
import json
import zipfile

import pytest

from experiments.research_extension.data import load_youtube_archive, save_artifact, sha256_json
from experiments.research_extension.schedule import (
    REQUIRED_ROLES,
    AllocationInfeasible,
    allocate_roles,
    build_schedule,
    provisional_feasibility,
)


def corpus(tmp_path, videos, *, bom=False):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, rows in videos.items():
            text = io.StringIO(newline="")
            writer = csv.writer(text)
            writer.writerow(["COMMENT_ID", "CONTENT", "CLASS"])
            writer.writerows(rows)
            archive.writestr(name + ".csv", text.getvalue().encode("utf-8-sig" if bom else "utf-8"))
    return load_youtube_archive(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def simple_corpus(tmp_path, n=30):
    return corpus(
        tmp_path, {"video": [(f"id-{i}", f"distinct {i} café 🌎 ", i % 2) for i in range(n)]}
    )


def role_requests(count=2):
    return {
        role: {"count": count, "class_counts": {0: count // 2, 1: count - count // 2}}
        for role in REQUIRED_ROLES
    }


def test_utf8_bom_preserves_source_and_global_ids(tmp_path):
    inventory = corpus(
        tmp_path,
        {
            "v1": [("same", " café 🌎 \n", 0), ("same", "second", 0)],
            "v2": [("same", "third", 0), ("other", "unique", 1)],
        },
        bom=True,
    )
    assert inventory.records[0].text == " café 🌎 \n"
    assert len({row.source_id for row in inventory.records}) == 4
    assert len({row.group_id for row in inventory.records[:3]}) == 1
    assert sum(not row.exclusions for row in inventory.records[:3]) == 1
    assert all(member["initial_bom"] for member in inventory.manifest["members"])
    assert inventory.manifest["video_holdout"] is False


def test_checksum_and_strict_utf8(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("v.csv", b"COMMENT_ID,CONTENT,CLASS\na,\xff,0\n")
    with pytest.raises(ValueError, match="checksum"):
        load_youtube_archive(path)
    with pytest.raises(UnicodeDecodeError):
        load_youtube_archive(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_transitive_links_conflicts_and_exclusions(tmp_path):
    inventory = corpus(
        tmp_path,
        {
            "a": [
                ("x", "first", 0),
                ("x", "linked", 0),
                ("y", "linked", 1),
                ("", "missing id", 0),
                ("z", "  ", 0),
                ("q", "bad label", 2),
            ]
        },
    )
    assert len({row.group_id for row in inventory.records[:3]}) == 1
    assert all("conflicting_group_labels" in row.exclusions for row in inventory.records[:3])
    assert len(inventory.eligible_records) == 0
    assert inventory.manifest["counts"]["excluded_by_reason"]["conflicting_group_labels"] == 3
    assert {link["kind"] for link in inventory.manifest["group_links"]} >= {
        "comment_id",
        "exact_text",
    }


def test_conservative_template_groups_not_input_rewrites(tmp_path):
    first = "Please take a moment to visit our music channel and hear this wonderful new song at https://one.example/a"
    second = "Please take a moment to visit our music channel and hear this wonderful new song at https://two.example/b"
    inventory = corpus(
        tmp_path,
        {"a": [("1", first, 1), ("2", second, 1), ("3", "nice video", 0), ("4", "NICE VIDEO", 0)]},
    )
    assert inventory.records[0].group_id == inventory.records[1].group_id
    assert inventory.records[2].group_id != inventory.records[3].group_id
    assert [row.text for row in inventory.records[:2]] == [first, second]
    assert len(inventory.eligible_records) == 4


def test_artifact_atomic_idempotence_and_incompatible_refusal(tmp_path):
    path = tmp_path / "artifacts" / "inventory.json"
    save_artifact(path, {"text": "café", "labels": [0]})
    save_artifact(path, {"labels": [0], "text": "café"})
    with pytest.raises(ValueError, match="incompatible overwrite"):
        save_artifact(path, {"labels": [1]})
    assert json.loads(path.read_text()) == {"text": "café", "labels": [0]}
    assert not list(path.parent.glob(".inventory.json.*"))


def test_roles_exact_fixed_group_disjoint_and_development_required(tmp_path):
    inventory = simple_corpus(tmp_path)
    requests = role_requests()
    allocation = allocate_roles(inventory, requests, seed=7)
    assert allocation == allocate_roles(inventory, requests, seed=7)
    used = [group for role in allocation["roles"].values() for group in role["group_ids"]]
    assert len(used) == len(set(used))
    assert all(
        role["count"] == 2 and role["class_counts"] == {"0": 1, "1": 1}
        for role in allocation["roles"].values()
    )
    assert allocation["video_holdout"] is False
    del requests["development"]
    with pytest.raises(ValueError, match="development"):
        allocate_roles(inventory, requests)


def test_no_silent_downsize_on_total_or_class_infeasibility(tmp_path):
    inventory = simple_corpus(tmp_path, 12)
    with pytest.raises(AllocationInfeasible) as error:
        allocate_roles(inventory, role_requests())
    assert error.value.report["available"]["examples"] == 12
    assert error.value.report["requested"]["held_out_evaluation"]["count"] == 2
    inventory = simple_corpus(tmp_path, 30)
    requests = {role: {"count": 4, "class_counts": {0: 4}} for role in REQUIRED_ROLES}
    with pytest.raises(AllocationInfeasible) as error:
        allocate_roles(inventory, requests)
    assert error.value.report["available"]["class_counts"]["0"] == 15


def test_group_indivisibility_infeasibility_not_greedy_failure(tmp_path):
    rows = []
    for i in range(8):
        for j in range(2):
            rows.append(
                (
                    f"id-{i}-{j}",
                    f"Please take a moment and listen to this wonderful song number {i} at https://site.example/{j}",
                    0,
                )
            )
    inventory = corpus(tmp_path, {"v": rows})
    requests = {role: {"count": 1} for role in REQUIRED_ROLES}
    with pytest.raises(AllocationInfeasible) as error:
        allocate_roles(inventory, requests)
    assert "no exact simultaneous" in str(error.value)
    assert error.value.report["group_size_gcd"] == 2
    requests = {role: {"count": 2} for role in REQUIRED_ROLES}
    allocation = allocate_roles(inventory, requests)
    assert all(role["count"] == 2 for role in allocation["roles"].values())


def test_schedule_no_group_reuse_and_paired_label_control(tmp_path):
    inventory = simple_corpus(tmp_path)
    requests = role_requests()
    requests["held_out_evaluation"] = {"count": 4, "class_counts": {0: 2, 1: 2}}
    allocation = allocate_roles(inventory, requests)
    windows = [
        {"role": "reference", "count": 2},
        {"role": "held_out_evaluation", "count": 2, "class_counts": {0: 1, 1: 1}},
        {"role": "held_out_evaluation", "count": 2, "class_counts": {0: 1, 1: 1}},
    ]
    schedule = build_schedule(inventory, allocation, windows, seed=2, schedule_id="stable")
    assert schedule == build_schedule(inventory, allocation, windows, seed=2, schedule_id="stable")
    group_ids = [group for window in schedule["windows"] for group in window["group_ids"]]
    assert len(group_ids) == len(set(group_ids))
    variants = {
        sample["source_id"]: {"label": 1 - sample["label"]}
        for window in schedule["windows"]
        for sample in window["samples"]
    }
    paired = build_schedule(
        inventory,
        allocation,
        windows,
        seed=99,
        schedule_id="label-control",
        paired_with=schedule,
        reuse_declaration={
            "kind": "label_only",
            "dependence": "identical source windows; reuse saved inference",
        },
        variants=variants,
        transformation={"labels": "binary complement", "text": "identity"},
    )
    assert paired["membership_sha256"] == schedule["membership_sha256"]
    assert paired["text_sha256"] == schedule["text_sha256"]
    assert paired["labels_sha256"] != schedule["labels_sha256"]
    with pytest.raises(ValueError, match="reuse"):
        build_schedule(
            inventory, allocation, windows, seed=2, schedule_id="bad", paired_with=schedule
        )
    with pytest.raises(AllocationInfeasible):
        build_schedule(
            inventory, allocation, windows + [windows[1]], seed=2, schedule_id="oversubscribed"
        )


def test_tampered_role_membership_refused_and_provisional_not_frozen(tmp_path):
    inventory = simple_corpus(tmp_path)
    allocation = allocate_roles(inventory, role_requests())
    allocation["roles"]["reference"]["source_ids"].pop()
    # Even a recomputed outer checksum cannot make a split group valid.
    allocation["allocation_sha256"] = sha256_json(
        {key: value for key, value in allocation.items() if key != "allocation_sha256"}
    )
    with pytest.raises(ValueError, match="membership"):
        build_schedule(
            inventory,
            allocation,
            [{"role": "reference", "count": 1}],
            seed=1,
            schedule_id="tampered",
        )
    provisional = provisional_feasibility(inventory)
    assert provisional["status"] == "infeasible"
    assert provisional["provisional_total"] == 1500
    assert provisional["frozen"] is False
    assert provisional["development"]["required"] is True
    assert "roles" not in provisional
