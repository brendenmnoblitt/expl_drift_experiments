"""Local-first, byte-pinned AG News inventory; no model evaluation or protocol freeze.

Run with --source-dir containing data/{train,test}-00000-of-00001.parquet,
 or explicitly --download. Only --requests allocates roles, via the generic engine.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .data import Inventory, SourceRecord, _Components, _template, save_artifact, sha256_json
from .schedule import REQUIRED_ROLES

DATASET_ID = "fancyzhx/ag_news"
REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
SOURCE_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
SOURCE_SHA256 = {
    "train": "fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee",
    "test": "71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb",
}
TASK_SCHEMA = {
    "version": 1,
    "candidate_ids": ["world", "sports", "business", "sci_tech"],
    "labels": {"0": "world", "1": "sports", "2": "business", "3": "sci_tech"},
    "reference_candidate": "world",
}
TRANSFORMS = {
    "version": 1,
    "input": "identity: exact parquet text; no stripping, case folding or Unicode normalization",
    "decode": "verified parquet bytes decoded by pyarrow; text must be a string",
    "grouping_only": (
        "global exact text and conservative data._template casefold/whitespace/URL keys"
    ),
    "deduplication": "first train-then-test/member/row eligible representative per exact text",
    "exclusions": (
        "invalid labels and blank text excluded; all conflicting-label group records excluded; "
        "every test record in a group containing any train record excluded, "
        "even if train record is otherwise excluded"
    ),
    "source_split": (
        "eligible whole groups have exactly one original source split; "
        "training uses train, every other required role uses test"
    ),
    "video": "not applicable; empty string, never a holdout variable",
}


def source_url(split: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/{SOURCE_FILES[split]}"


def load_ag_news(
    source_dir: str | Path, *, fixture_sha256: Mapping[str, str] | None = None
) -> Inventory:
    """Verify both files before decoding; explicit custom pins always mark fixture provenance."""
    pins = dict(SOURCE_SHA256 if fixture_sha256 is None else fixture_sha256)
    if set(pins) != set(SOURCE_FILES):
        raise ValueError("source pins must contain exactly train and test")
    blobs = {}
    for split, member in SOURCE_FILES.items():
        raw = (Path(source_dir) / member).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != pins[split]:
            raise ValueError(f"{split} checksum mismatch: expected {pins[split]}, got {digest}")
        blobs[split] = raw
    import pyarrow.parquet as pq

    files = [
        {"source_split": split, "member": member, "sha256": pins[split]}
        for split, member in SOURCE_FILES.items()
    ]
    composite = sha256_json({SOURCE_FILES[split]: pins[split] for split in SOURCE_FILES})
    rows: list[dict[str, Any]] = []
    for split, member in SOURCE_FILES.items():
        table = pq.read_table(io.BytesIO(blobs[split]))
        if table.column_names.count("text") != 1 or table.column_names.count("label") != 1:
            raise ValueError(f"{member}: required unique text/label columns missing")
        for number, item in enumerate(table.select(["text", "label"]).to_pylist(), start=1):
            text, value = item["text"], item["label"]
            if not isinstance(text, str):
                raise ValueError(f"{member} row {number}: text must be a string")
            label = value if type(value) is int and value in range(4) else None
            reasons = []
            if label is None:
                reasons.append("invalid_label")
            if not text.strip():
                reasons.append("empty_or_whitespace_text")
            rows.append(
                dict(
                    source_id="ag:" + sha256_json([composite, split, member, number]),
                    comment_id="",
                    video="",
                    member=member,
                    row_number=number,
                    text=text,
                    label=label,
                    source_label=str(value),
                    source_split=split,
                    exclusions=reasons,
                )
            )
    if not rows:
        raise ValueError("source files contain no records")
    if fixture_sha256 is None:
        counts = Counter(row["source_split"] for row in rows)
        if counts != {"train": 120000, "test": 7600}:
            raise ValueError(f"public pinned source row counts mismatch: {dict(counts)}")
    grouped = _Components(len(rows))
    exact, templates = {}, {}
    links = []
    for i, row in enumerate(rows):
        for kind, key, seen in (
            ("exact_text", row["text"], exact),
            ("template", _template(row["text"]), templates),
        ):
            if key is None:
                continue
            if key in seen:
                other = seen[key]
                grouped.join(i, other)
                links.append(
                    {
                        "kind": kind,
                        "source_ids": [rows[other]["source_id"], row["source_id"]],
                        "key_sha256": sha256_json(key),
                    }
                )
            else:
                seen[key] = i
    clusters = defaultdict(list)
    for i in range(len(rows)):
        clusters[grouped.root(i)].append(i)
    groups = []
    for indices in clusters.values():
        ids = sorted(rows[i]["source_id"] for i in indices)
        group_id = "ag-group:" + sha256_json(ids)
        labels = sorted({rows[i]["label"] for i in indices if rows[i]["label"] is not None})
        conflict = len(labels) > 1
        splits = sorted({rows[i]["source_split"] for i in indices})
        representatives = set()
        for i in indices:
            row = rows[i]
            row["group_id"] = group_id
            if conflict:
                row["exclusions"].append("conflicting_group_labels")
            if row["source_split"] == "test" and "train" in splits:
                row["exclusions"].append("test_group_overlaps_training")
            if not row["exclusions"]:
                if row["text"] in representatives:
                    row["exclusions"].append("duplicate_exact_text")
                else:
                    representatives.add(row["text"])
        eligible = [rows[i] for i in indices if not rows[i]["exclusions"]]
        groups.append(
            {
                "group_id": group_id,
                "source_ids": ids,
                "eligible_source_ids": [row["source_id"] for row in eligible],
                "labels": labels,
                "conflicting_labels": conflict,
                "source_splits": splits,
                "eligible_source_splits": sorted({row["source_split"] for row in eligible}),
                "videos": [],
            }
        )
    groups.sort(key=lambda group: group["group_id"])
    records = tuple(SourceRecord(**{**row, "exclusions": tuple(row["exclusions"])}) for row in rows)
    eligible = [row for row in records if not row.exclusions]
    manifest = {
        "schema_version": 2,
        "dataset": "AG News",
        "dataset_id": DATASET_ID,
        "revision": REVISION if fixture_sha256 is None else None,
        "provenance": "public-pinned" if fixture_sha256 is None else "fixture/explicit-pin",
        "archive_sha256": composite,
        "members": files,
        "source_urls": {split: source_url(split) for split in SOURCE_FILES}
        if fixture_sha256 is None
        else {},
        "task_schema": TASK_SCHEMA,
        "transforms": TRANSFORMS,
        "video_holdout": False,
        "role_source_splits": {
            role: "train" if role == "training" else "test" for role in REQUIRED_ROLES
        },
        "group_links": links,
        "hashes": {
            "source_ids": sha256_json([row.source_id for row in records]),
            "source_files": sha256_json(files),
            "source_splits": sha256_json([[row.source_id, row.source_split] for row in records]),
            "text": sha256_json([[row.source_id, row.text] for row in records]),
            "labels": sha256_json([[row.source_id, row.source_label] for row in records]),
            "groups": sha256_json(groups),
            "group_links": sha256_json(links),
            "task_schema": sha256_json(TASK_SCHEMA),
            "transforms": sha256_json(TRANSFORMS),
            "exclusions": sha256_json([[row.source_id, row.exclusions] for row in records]),
        },
        "counts": {
            "raw": len(records),
            "eligible": len(eligible),
            "eligible_groups": sum(bool(group["eligible_source_ids"]) for group in groups),
            "by_class": {
                str(label): sum(row.label == label for row in eligible) for label in range(4)
            },
            "by_source_split": {
                split: sum(row.source_split == split for row in eligible) for split in SOURCE_FILES
            },
            "by_source_split_class": {
                split: {
                    str(label): sum(
                        row.source_split == split and row.label == label for row in eligible
                    )
                    for label in range(4)
                }
                for split in SOURCE_FILES
            },
            "excluded_by_reason": dict(
                Counter(reason for row in records for reason in row.exclusions)
            ),
        },
    }
    manifest["inventory_sha256"] = sha256_json(
        {"manifest": manifest, "records": [asdict(row) for row in records], "groups": groups}
    )
    return Inventory(records, tuple(groups), manifest)


def download_ag_news(source_dir: str | Path) -> None:
    """Download pinned bytes through temporary files, without replacing differing local bytes."""
    for split, member in SOURCE_FILES.items():
        target = Path(source_dir) / member
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != SOURCE_SHA256[split]:
                raise ValueError(f"existing {target} checksum mismatch")
            continue
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with urlopen(source_url(split), timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                stream.flush()
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != SOURCE_SHA256[split]:
                raise ValueError(f"downloaded {split} checksum mismatch")
            import os

            try:
                os.link(temporary, target)
            except FileExistsError:
                if hashlib.sha256(target.read_bytes()).hexdigest() != SOURCE_SHA256[split]:
                    raise ValueError(f"existing {target} checksum mismatch")
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", type=Path)
    source.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--schedule-id")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.windows and (not args.requests or not args.schedule_id):
        parser.error("--windows requires --requests and --schedule-id")
    if args.download:
        with tempfile.TemporaryDirectory(prefix="ag-news-") as directory:
            download_ag_news(directory)
            inventory = load_ag_news(directory)
    else:
        inventory = load_ag_news(args.source_dir)
    save_artifact(args.output / "inventory.json", inventory.artifact())
    if args.requests:
        from .schedule import AllocationInfeasible, allocate_roles, build_schedule

        try:
            allocation = allocate_roles(
                inventory, json.loads(args.requests.read_text(encoding="utf-8")), seed=args.seed
            )
        except AllocationInfeasible as error:
            save_artifact(args.output / "allocation-infeasibility.json", error.report)
            raise SystemExit(json.dumps(error.report, sort_keys=True)) from error
        save_artifact(args.output / "roles.json", allocation)
        if args.windows:
            schedule = build_schedule(
                inventory,
                allocation,
                json.loads(args.windows.read_text(encoding="utf-8")),
                seed=args.seed,
                schedule_id=args.schedule_id,
            )
            save_artifact(args.output / "schedule.json", schedule)
    print(
        json.dumps(
            {
                "inventory": str(args.output / "inventory.json"),
                "counts": inventory.manifest["counts"],
                "protocol_frozen": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
