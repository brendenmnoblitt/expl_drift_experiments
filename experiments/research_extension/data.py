"""CPU-only, source-preserving inventory of the pinned YouTube corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

DATASET_URL = "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip"
DATASET_SHA256 = "bd6182891adb3cfc8334b82c062176dfbebc563bf0ba07e31c2645f916865a0a"
TASK_SCHEMA = {
    "version": 1,
    "candidate_ids": ["ham", "spam"],
    "labels": {"0": "ham", "1": "spam"},
    "reference_candidate": "ham",
}
TRANSFORMS = {
    "version": 1,
    "input": "identity: exact CONTENT field; no stripping, case folding or Unicode normalization",
    "decode": (
        "strict UTF-8; optional initial CSV BOM removed by utf-8-sig; "
        "CSV newline semantics preserved"
    ),
    "grouping_only": (
        "same nonempty COMMENT_ID globally; exact CONTENT; casefold/whitespace template "
        "for >=40 characters and >=8 alphabetic words; URL replacement only for >=80 "
        "characters and >=12 alphabetic words"
    ),
    "deduplication": (
        "one first archive-member/row representative per transitive same-ID/exact-text "
        "component; entire template group excluded on conflicting valid labels"
    ),
    "video": "retained grouping/uncertainty variable, NOT video-disjoint holdout",
}


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    comment_id: str
    video: str
    member: str
    row_number: int
    text: str
    label: int | None
    source_label: str
    group_id: str
    exclusions: tuple[str, ...]


@dataclass(frozen=True)
class Inventory:
    records: tuple[SourceRecord, ...]
    groups: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]

    @property
    def eligible_records(self) -> tuple[SourceRecord, ...]:
        return tuple(record for record in self.records if not record.exclusions)

    def artifact(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "records": [asdict(row) for row in self.records],
            "groups": list(self.groups),
        }


class _Components:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def root(self, n: int) -> int:
        while self.parent[n] != n:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def join(self, a: int, b: int) -> None:
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def _template(text: str) -> str | None:
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    if len(text) < 40 or len(words) < 8:
        return None
    key = text.casefold()
    if len(text) >= 80 and len(words) >= 12:
        key = re.sub(r"https?://[^\s<>]+|www\.[^\s<>]+", "<url>", key)
    return " ".join(key.split())


def load_youtube_archive(path: str | Path, *, expected_sha256: str = DATASET_SHA256) -> Inventory:
    """Check bytes before strict decoding. Custom pins are for explicit fixtures/corpora."""
    archive = Path(path).read_bytes()
    digest = hashlib.sha256(archive).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"archive checksum mismatch: expected {expected_sha256}, got {digest}")
    rows: list[dict[str, Any]] = []
    members = []
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = [
            n for n in bundle.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX/")
        ]
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate CSV member paths")
        for member in sorted(names):
            raw = bundle.read(member)
            decoded = raw.decode("utf-8-sig", errors="strict")
            reader = csv.DictReader(io.StringIO(decoded, newline=""))
            if (
                reader.fieldnames is None
                or len(reader.fieldnames) != len(set(reader.fieldnames))
                or not {"COMMENT_ID", "CONTENT", "CLASS"}.issubset(reader.fieldnames)
            ):
                raise ValueError(f"invalid CSV header in {member}")
            members.append(
                {
                    "member": member,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "initial_bom": raw.startswith(b"\xef\xbb\xbf"),
                }
            )
            for number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"malformed CSV record in {member}, logical row {number}")
                source_label = row["CLASS"]
                label = int(source_label) if source_label in ("0", "1") else None
                reasons = []
                if not row["COMMENT_ID"].strip():
                    reasons.append("missing_comment_id")
                if not row["CONTENT"].strip():
                    reasons.append("empty_or_whitespace_text")
                if label is None:
                    reasons.append("invalid_label")
                source_id = "yt:" + sha256_json([digest, member, number])
                rows.append(
                    {
                        "source_id": source_id,
                        "comment_id": row["COMMENT_ID"],
                        "video": Path(member).stem,
                        "member": member,
                        "row_number": number,
                        "text": row["CONTENT"],
                        "label": label,
                        "source_label": source_label,
                        "exclusions": reasons,
                    }
                )
    if not rows:
        raise ValueError("archive contains no CSV source records")
    if digest == DATASET_SHA256 and len(rows) != 1956:
        raise ValueError(f"pinned corpus must contain 1956 source records, found {len(rows)}")
    exact = _Components(len(rows))
    grouped = _Components(len(rows))
    links: list[dict[str, Any]] = []
    by_id: dict[str, int] = {}
    by_text: dict[str, int] = {}
    by_template: dict[str, int] = {}
    for index, row in enumerate(rows):
        for kind, key, seen in (
            ("comment_id", row["comment_id"] if row["comment_id"].strip() else None, by_id),
            ("exact_text", row["text"], by_text),
            ("template", _template(row["text"]), by_template),
        ):
            if key is None:
                continue
            if key in seen:
                other = seen[key]
                grouped.join(index, other)
                if kind != "template":
                    exact.join(index, other)
                links.append(
                    {
                        "kind": kind,
                        "source_ids": [rows[other]["source_id"], row["source_id"]],
                        "key_sha256": sha256_json(key),
                    }
                )
            else:
                seen[key] = index
    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        clusters[grouped.root(index)].append(index)
    groups = []
    for indices in clusters.values():
        source_ids = sorted(rows[i]["source_id"] for i in indices)
        group_id = "yt-group:" + sha256_json(source_ids)
        labels = sorted({rows[i]["label"] for i in indices if rows[i]["label"] is not None})
        conflict = len(labels) > 1
        representatives: dict[int, int] = {}
        for i in indices:
            row = rows[i]
            row["group_id"] = group_id
            if conflict:
                row["exclusions"].append("conflicting_group_labels")
            if not row["exclusions"]:
                component = exact.root(i)
                if component in representatives:
                    row["exclusions"].append("duplicate_source_or_exact_text")
                else:
                    representatives[component] = i
        groups.append(
            {
                "group_id": group_id,
                "source_ids": source_ids,
                "eligible_source_ids": [
                    rows[i]["source_id"] for i in indices if not rows[i]["exclusions"]
                ],
                "labels": labels,
                "conflicting_labels": conflict,
                "videos": sorted({rows[i]["video"] for i in indices}),
            }
        )
    records = tuple(SourceRecord(**{**row, "exclusions": tuple(row["exclusions"])}) for row in rows)
    groups.sort(key=lambda group: group["group_id"])
    eligible = [row for row in records if not row.exclusions]
    manifest = {
        "schema_version": 1,
        "dataset": "UCI YouTube Spam Collection",
        "archive_sha256": digest,
        "url": DATASET_URL if digest == DATASET_SHA256 else None,
        "license": "CC BY 4.0" if digest == DATASET_SHA256 else "fixture/explicit-pin",
        "members": members,
        "task_schema": TASK_SCHEMA,
        "transforms": TRANSFORMS,
        "video_holdout": False,
        "hashes": {
            "source_ids": sha256_json([row.source_id for row in records]),
            "text": sha256_json([[row.source_id, row.text] for row in records]),
            "labels": sha256_json([[row.source_id, row.source_label] for row in records]),
            "groups": sha256_json(groups),
            "transforms": sha256_json(TRANSFORMS),
            "task_schema": sha256_json(TASK_SCHEMA),
            "exclusions": sha256_json([[row.source_id, row.exclusions] for row in records]),
            "group_links": sha256_json(links),
        },
        "counts": {
            "raw": len(records),
            "eligible": len(eligible),
            "eligible_groups": sum(bool(group["eligible_source_ids"]) for group in groups),
            "by_class": dict(Counter(str(row.label) for row in eligible)),
            "by_video": dict(Counter(row.video for row in eligible)),
            "by_video_class": {
                video: dict(Counter(str(row.label) for row in eligible if row.video == video))
                for video in sorted({row.video for row in records})
            },
            "excluded_by_reason": dict(
                Counter(reason for row in records for reason in row.exclusions)
            ),
        },
        "group_links": links,
    }
    manifest["inventory_sha256"] = sha256_json(
        {"manifest": manifest, "records": [asdict(row) for row in records], "groups": groups}
    )
    return Inventory(records, tuple(groups), manifest)


def save_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    """Atomic creation, idempotent equality, and no incompatible replacement (even races)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text(encoding="utf-8")) != json.loads(payload):
                raise ValueError(f"refusing incompatible overwrite of {path}")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="local zip with the public pinned checksum")
    source.add_argument(
        "--download", action="store_true", help="explicitly download the pinned public UCI archive"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--requests",
        type=Path,
        help="JSON role mapping with count and optional class_counts; includes development",
    )
    parser.add_argument(
        "--windows",
        type=Path,
        help="JSON list of window requests; requires --requests and --schedule-id",
    )
    parser.add_argument("--schedule-id")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.windows and (not args.requests or not args.schedule_id):
        parser.error("--windows requires --requests and --schedule-id")
    if args.download:
        with urlopen(DATASET_URL, timeout=60) as response:
            archive = response.read()
        if hashlib.sha256(archive).hexdigest() != DATASET_SHA256:
            raise ValueError("downloaded archive does not match the public pin")
        with tempfile.NamedTemporaryFile(suffix=".zip") as temporary:
            temporary.write(archive)
            temporary.flush()
            inventory = load_youtube_archive(temporary.name)
    else:
        inventory = load_youtube_archive(args.archive)
    from .schedule import (
        AllocationInfeasible,
        allocate_roles,
        build_schedule,
        provisional_feasibility,
    )

    save_artifact(args.output / "inventory.json", inventory.artifact())
    feasibility = provisional_feasibility(inventory, seed=args.seed)
    save_artifact(args.output / "provisional-feasibility.json", feasibility)
    if args.requests:
        requests = json.loads(args.requests.read_text(encoding="utf-8"))
        try:
            allocation = allocate_roles(inventory, requests, seed=args.seed)
        except AllocationInfeasible as error:
            save_artifact(args.output / "allocation-infeasibility.json", error.report)
            raise SystemExit(json.dumps(error.report, sort_keys=True)) from error
        save_artifact(args.output / "roles.json", allocation)
        if args.windows:
            windows = json.loads(args.windows.read_text(encoding="utf-8"))
            schedule = build_schedule(
                inventory, allocation, windows, seed=args.seed, schedule_id=args.schedule_id
            )
            save_artifact(args.output / "schedule.json", schedule)
    print(
        json.dumps(
            {
                "inventory": str(args.output / "inventory.json"),
                "counts": inventory.manifest["counts"],
                "provisional": feasibility["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
