"""Persisted corpus integrity and role-only task-training admission boundaries."""

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from experiments.research_extension.data import Inventory, SourceRecord, sha256_json
from experiments.research_extension.schedule import REQUIRED_ROLES, allocate_roles
from experiments.research_extension.training import prepare_training_data, train_reference


def inventory_fixture(labels=(0, 1, 2, 3)):
    records = []
    for split, copies in (("train", 1), ("test", 6)):
        for label in labels:
            for copy in range(copies):
                source = f"{split}-{label}-{copy}"
                records.append(
                    SourceRecord(
                        source_id=source,
                        comment_id="",
                        video="",
                        member=split,
                        row_number=len(records) + 1,
                        text=f"Original input {source}\n",
                        label=label,
                        source_label=str(label),
                        group_id=source,
                        exclusions=(),
                        source_split=split,
                    )
                )
    schema = {
        "candidate_ids": [f"class-{label}" for label in labels],
        "labels": {str(label): f"class-{label}" for label in labels},
        "reference_candidate": f"class-{labels[0]}",
    }
    manifest = {
        "schema_version": 2,
        "archive_sha256": sha256_json("fixture source"),
        "task_schema": schema,
        "transforms": {"input": "identity"},
        "hashes": {
            "task_schema": sha256_json(schema),
            "transforms": sha256_json({"input": "identity"}),
        },
        "role_source_splits": {
            role: "train" if role == "training" else "test" for role in REQUIRED_ROLES
        },
    }
    groups = tuple({"group_id": row.group_id} for row in records)
    manifest["inventory_sha256"] = sha256_json(
        {"manifest": manifest, "records": [asdict(row) for row in records], "groups": groups}
    )
    return Inventory(tuple(records), groups, manifest)


def balanced_requests(labels=(0, 1, 2, 3)):
    return {
        role: {"count": len(labels), "class_counts": {str(label): 1 for label in labels}}
        for role in REQUIRED_ROLES
    }


def test_restored_inventory_retains_source_split_and_training_excludes_reserved_roles():
    original = inventory_fixture()
    restored = Inventory.from_artifact(json.loads(json.dumps(original.artifact())))
    allocation = allocate_roles(restored, balanced_requests())
    data = prepare_training_data(restored, allocation)
    assert {row.source_split for row in data["training"]} == {"train"}
    assert {row.source_split for row in data["development"]} == {"test"}
    reserved = {
        source
        for role, value in allocation["roles"].items()
        if role not in ("training", "development")
        for source in value["source_ids"]
    }
    assert reserved.isdisjoint(row.source_id for row in data["training"])
    assert reserved.isdisjoint(row.source_id for row in data["development"])
    assert {row.label for row in data["training"]} == {0, 1, 2, 3}


def test_restored_inventory_rejects_changed_input_text_and_obsolete_schema():
    artifact = json.loads(json.dumps(inventory_fixture().artifact()))
    changed = deepcopy(artifact)
    changed["records"][0]["text"] = "rewritten model input"
    with pytest.raises(ValueError, match="content/hash"):
        Inventory.from_artifact(changed)
    artifact["manifest"]["schema_version"] = 1
    with pytest.raises(ValueError, match="regenerate"):
        Inventory.from_artifact(artifact)


def test_training_rejects_nondense_classifier_mapping_before_loading_weights(tmp_path):
    inventory = inventory_fixture(labels=(1, 2, 3, 4))
    allocation = allocate_roles(inventory, balanced_requests(labels=(1, 2, 3, 4)))
    with pytest.raises(ValueError, match="dense"):
        train_reference(
            inventory,
            allocation,
            tmp_path / "missing-base",
            tmp_path / "output",
            expected_base_sha256="a" * 64,
        )
    assert not (tmp_path / "output").exists()


def test_training_refuses_existing_outputs_before_loading_or_overwriting_weights(tmp_path):
    inventory = inventory_fixture()
    allocation = allocate_roles(inventory, balanced_requests())
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "previous-result"
    marker.write_text("retained")
    with pytest.raises(FileExistsError, match="fresh"):
        train_reference(
            inventory, allocation, tmp_path / "missing-base", output, expected_base_sha256="a" * 64
        )
    assert marker.read_text() == "retained"
