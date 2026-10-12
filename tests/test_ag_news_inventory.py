"""Behavioral fixtures for pinned, source-preserving AG News inventories."""

import hashlib

import pytest

from experiments.research_extension.ag_news import SOURCE_FILES, SOURCE_SHA256, load_ag_news
from experiments.research_extension.data import Inventory, save_artifact
from experiments.research_extension.schedule import REQUIRED_ROLES, allocate_roles


def fixture_files(tmp_path, train, test):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pins = {}
    for split, rows in (("train", train), ("test", test)):
        path = tmp_path / SOURCE_FILES[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({"text": [text for text, _ in rows], "label": [label for _, label in rows]}),
            path,
        )
        pins[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pins


def corpus(tmp_path, train, test):
    return load_ag_news(tmp_path, fixture_sha256=fixture_files(tmp_path, train, test))


def test_cross_split_exact_and_template_groups(tmp_path):
    long = "A long example with enough alphabetic words for conservative template grouping today"
    inventory = corpus(
        tmp_path,
        [("exact", 0), (long, 1), (long, 1)],
        [("exact", 0), (long.upper(), 1), ("independent", 2)],
    )
    rows = inventory.records
    assert rows[0].group_id == rows[3].group_id
    assert rows[1].group_id == rows[2].group_id == rows[4].group_id
    assert rows[1].text == long and rows[4].text == long.upper()
    assert "duplicate_exact_text" in rows[2].exclusions
    assert all("test_group_overlaps_training" in rows[i].exclusions for i in (3, 4))
    assert {row.source_id for row in inventory.eligible_records} == {
        rows[i].source_id for i in (0, 1, 5)
    }
    assert all(len(group["eligible_source_splits"]) <= 1 for group in inventory.groups)


def test_conflict_excludes_entire_template_group(tmp_path):
    text = "This sufficiently long article has more than eight alphabetic words in its headline"
    inventory = corpus(tmp_path, [(text, 0)], [(text.upper(), 3), ("valid", 2)])
    assert all("conflicting_group_labels" in row.exclusions for row in inventory.records[:2])
    assert "test_group_overlaps_training" in inventory.records[1].exclusions
    assert [row.text for row in inventory.eligible_records] == ["valid"]


def test_unicode_identity_and_immutable_roundtrip(tmp_path):
    texts = [" café 🌎 \n", "cafe\u0301", "café", "界\t"]
    inventory = corpus(tmp_path, [(text, i) for i, text in enumerate(texts)], [("test", 0)])
    assert [row.text for row in inventory.records[:4]] == texts
    assert len({row.group_id for row in inventory.records[:4]}) == 4
    assert all(row.video == "" and row.comment_id == "" for row in inventory.records)
    assert inventory.records[0].row_number == 1
    assert inventory.records[0].member == SOURCE_FILES["train"]
    assert inventory.manifest["provenance"] == "fixture/explicit-pin"
    assert inventory.manifest["revision"] is None
    assert Inventory.from_artifact(inventory.artifact()) == inventory
    path = tmp_path / "inventory.json"
    save_artifact(path, inventory.artifact())
    save_artifact(path, inventory.artifact())
    with pytest.raises(ValueError, match="overwrite"):
        save_artifact(path, {"different": True})


def test_invalid_class_and_checksum(tmp_path):
    pins = fixture_files(tmp_path, [("invalid", 4), ("blank valid", 0)], [("test", 1)])
    inventory = load_ag_news(tmp_path, fixture_sha256=pins)
    assert inventory.records[0].label is None
    assert inventory.records[0].source_label == "4"
    assert "invalid_label" in inventory.records[0].exclusions
    with pytest.raises(ValueError, match="checksum"):
        load_ag_news(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        load_ag_news(tmp_path, fixture_sha256={**pins, "test": "0" * 64})
    with pytest.raises(ValueError, match="exactly train and test"):
        load_ag_news(tmp_path, fixture_sha256={"train": pins["train"]})
    assert (
        SOURCE_SHA256["train"] == "fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee"
    )
    assert (
        SOURCE_SHA256["test"] == "71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb"
    )


def test_any_training_record_blocks_test_and_test_dedup(tmp_path):
    inventory = corpus(tmp_path, [("overlap", 99)], [("overlap", 1), ("repeat", 2), ("repeat", 2)])
    assert "test_group_overlaps_training" in inventory.records[1].exclusions
    assert not inventory.records[2].exclusions
    assert "duplicate_exact_text" in inventory.records[3].exclusions


def test_four_class_role_preservation(tmp_path):
    train = [(f"train {label}", label) for label in range(4)]
    test = [
        (f"test {role} {label}", label)
        for role in REQUIRED_ROLES
        if role != "training"
        for label in range(4)
    ]
    inventory = corpus(tmp_path, train, test)
    requests = {
        role: {"count": 4, "class_counts": {str(label): 1 for label in range(4)}}
        for role in REQUIRED_ROLES
    }
    allocation = allocate_roles(inventory, requests, seed=17)
    assert allocation == allocate_roles(inventory, requests, seed=17)
    by_id = {row.source_id: row for row in inventory.records}
    used = set()
    for role, entry in allocation["roles"].items():
        selected = [by_id[source_id] for source_id in entry["source_ids"]]
        assert {row.label for row in selected} == set(range(4))
        assert {row.source_split for row in selected} == {"train" if role == "training" else "test"}
        assert not used.intersection(entry["source_ids"])
        used.update(entry["source_ids"])
    assert inventory.manifest["counts"]["by_source_split_class"] == {
        "train": {str(label): 1 for label in range(4)},
        "test": {str(label): 6 for label in range(4)},
    }
    assert inventory.manifest["role_source_splits"] == {
        role: "train" if role == "training" else "test" for role in REQUIRED_ROLES
    }
