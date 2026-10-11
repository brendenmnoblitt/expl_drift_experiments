"""Exact whole-group allocations and immutable, role-preserving CPU schedules."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from math import gcd
from typing import Any

from .data import Inventory, sha256_json

REQUIRED_ROLES = (
    "development",
    "training",
    "reference",
    "calibration",
    "threshold_tuning",
    "held_out_stable",
    "held_out_evaluation",
)
PROVISIONAL_COUNTS = {
    "training": 300,
    "threshold_tuning": 200,
    "reference": 50,
    "calibration": 200,
    "held_out_stable": 250,
    "held_out_evaluation": 500,
}


class AllocationInfeasible(ValueError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(report["reason"])


def _requests(requests: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for name, request in requests.items():
        if not isinstance(name, str) or not name:
            raise ValueError("role/window names must be nonempty strings")
        count = request.get("count")
        if type(count) is not int or count < 0:
            raise ValueError(f"{name}: count must be a nonnegative integer")
        classes = request.get("class_counts")
        if classes is not None:
            normalized = {}
            for label, value in classes.items():
                key = str(label)
                if (
                    key not in ("0", "1")
                    or key in normalized
                    or type(value) is not int
                    or value < 0
                ):
                    raise ValueError(f"{name}: invalid/duplicate class count {label!r}")
                normalized[key] = value
            classes = {key: normalized.get(key, 0) for key in ("0", "1")}
            if sum(classes.values()) != count:
                raise ValueError(f"{name}: class counts must sum to count")
        result[name] = {"count": count, "class_counts": classes}
    return result


def _eligible_groups(inventory: Inventory) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for record in inventory.eligible_records:
        groups[record.group_id].append(record)
    return dict(groups)


def _partition(
    groups: Mapping[str, Sequence[Any]], requests: Mapping[str, Mapping[str, Any]], seed: int
) -> dict[str, list[str]]:
    """Exact bounded multiset search, compressing homogeneous groups by class/size.

    Unassigned groups are permitted. No timeout or heuristic failure is reported
    as infeasibility: failure follows exhaustive, memoized integer search.
    """
    requests = _requests(requests)
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for group_id, rows in sorted(groups.items()):
        labels = {row.label for row in rows}
        if len(labels) != 1 or next(iter(labels)) not in (0, 1):
            raise ValueError(f"eligible group {group_id} has inconsistent labels")
        buckets[(rows[0].label, len(rows))].append(group_id)
    types = sorted(buckets, key=lambda key: (-key[1], key[0]))
    initial = tuple(len(buckets[key]) for key in types)
    available = {
        str(label): sum(
            size * amount for (kind, size), amount in zip(types, initial) if kind == label
        )
        for label in (0, 1)
    }
    total = sum(available.values())
    report = {
        "status": "infeasible",
        "requested": requests,
        "available": {
            "examples": total,
            "class_counts": available,
            "groups": sum(initial),
            "group_sizes": {
                str(label): dict(
                    Counter(len(rows) for rows in groups.values() if rows[0].label == label)
                )
                for label in (0, 1)
            },
        },
        "whole_groups": True,
    }
    required_total = sum(request["count"] for request in requests.values())
    required_class = {
        key: sum(
            request["class_counts"][key]
            for request in requests.values()
            if request["class_counts"] is not None
        )
        for key in ("0", "1")
    }
    if required_total > total or any(required_class[key] > available[key] for key in available):
        raise AllocationInfeasible(
            {
                **report,
                "reason": (
                    f"requested {required_total} examples and constrained class totals "
                    f"{required_class}; only {total} examples with classes {available} "
                    "are eligible"
                ),
            }
        )
    # Tightly class-constrained and larger bins first reduce symmetric search.
    names = sorted(
        requests,
        key=lambda name: (requests[name]["class_counts"] is None, -requests[name]["count"], name),
    )
    failed: set[tuple[int, tuple[int, ...]]] = set()

    def subsets(remaining: tuple[int, ...], request: dict[str, Any]):
        suffix = [[0, 0] for _ in range(len(types) + 1)]
        for i in range(len(types) - 1, -1, -1):
            suffix[i] = suffix[i + 1].copy()
            label, size = types[i]
            suffix[i][label] += remaining[i] * size
        classes = request["class_counts"]
        targets = None if classes is None else (classes["0"], classes["1"])
        selected = [0] * len(types)

        def visit(i: int, need: int, needs: tuple[int, int] | None):
            if need == 0:
                yield tuple(selected[:i] + [0] * (len(types) - i))
                return
            if (
                i == len(types)
                or need > sum(suffix[i])
                or (needs is not None and any(needs[k] > suffix[i][k] for k in (0, 1)))
            ):
                return
            label, size = types[i]
            maximum = min(
                remaining[i], need // size, need // size if needs is None else needs[label] // size
            )
            for amount in range(maximum, -1, -1):
                selected[i] = amount
                next_needs = (
                    None
                    if needs is None
                    else tuple(needs[k] - (size * amount if k == label else 0) for k in (0, 1))
                )
                yield from visit(i + 1, need - size * amount, next_needs)
            selected[i] = 0

        yield from visit(0, request["count"], targets)

    def solve(index: int, remaining: tuple[int, ...]):
        if index == len(names):
            return []
        state = (index, remaining)
        if state in failed:
            return None
        future = [requests[name] for name in names[index:]]
        capacity = [
            sum(size * amount for (kind, size), amount in zip(types, remaining) if kind == label)
            for label in (0, 1)
        ]
        if sum(item["count"] for item in future) > sum(capacity) or any(
            sum(
                item["class_counts"][str(label)]
                for item in future
                if item["class_counts"] is not None
            )
            > capacity[label]
            for label in (0, 1)
        ):
            failed.add(state)
            return None
        for selection in subsets(remaining, requests[names[index]]):
            rest = tuple(amount - take for amount, take in zip(remaining, selection))
            solution = solve(index + 1, rest)
            if solution is not None:
                return [selection, *solution]
        failed.add(state)
        return None

    solution = solve(0, initial)
    if solution is None:
        divisor = 0
        for _, size in types:
            divisor = gcd(divisor, size)
        raise AllocationInfeasible(
            {
                **report,
                "reason": (
                    "no exact simultaneous whole-group allocation satisfies the "
                    "requested role/window and class counts"
                ),
                "exhaustive_failed_states": len(failed),
                "group_size_gcd": divisor,
            }
        )
    rng = random.Random(seed)
    pools = []
    for key in types:
        pool = buckets[key].copy()
        rng.shuffle(pool)
        pools.append(pool)
    offsets = [0] * len(types)
    allocated = {}
    for name, selection in zip(names, solution):
        selected = []
        for i, amount in enumerate(selection):
            selected.extend(pools[i][offsets[i] : offsets[i] + amount])
            offsets[i] += amount
        allocated[name] = sorted(selected)
    return {name: allocated[name] for name in requests}


def _seal(artifact: dict[str, Any], key: str) -> dict[str, Any]:
    return {**artifact, key: sha256_json(artifact)}


def allocate_roles(
    inventory: Inventory, requests: Mapping[str, Mapping[str, Any]], *, seed: int = 42
) -> dict[str, Any]:
    """Allocate every mandatory role; separate groups with no implicit overlap."""
    if set(requests) != set(REQUIRED_ROLES):
        raise ValueError(
            f"roles must be exactly {REQUIRED_ROLES}; development is mandatory "
            "and cannot be taken from held-out roles"
        )
    normalized = _requests(requests)
    if normalized["development"]["count"] == 0:
        raise ValueError("development must contain designated examples")
    groups = _eligible_groups(inventory)
    assigned = _partition(groups, normalized, seed)
    roles = {}
    for role, group_ids in assigned.items():
        rows = sorted(
            (row for group_id in group_ids for row in groups[group_id]),
            key=lambda row: row.source_id,
        )
        roles[role] = {
            "source_ids": [row.source_id for row in rows],
            "group_ids": group_ids,
            "count": len(rows),
            "class_counts": dict(Counter(str(row.label) for row in rows)),
            "videos": dict(Counter(row.video for row in rows)),
        }
    return _seal(
        {
            "schema_version": 1,
            "inventory_sha256": inventory.manifest["inventory_sha256"],
            "archive_sha256": inventory.manifest["archive_sha256"],
            "task_schema_sha256": inventory.manifest["hashes"]["task_schema"],
            "transforms_sha256": inventory.manifest["hashes"]["transforms"],
            "seed": seed,
            "requests": normalized,
            "roles": roles,
            "video_holdout": False,
            "permitted_role_overlap": [],
            "membership_sha256": sha256_json(
                {role: data["source_ids"] for role, data in roles.items()}
            ),
        },
        "allocation_sha256",
    )


def provisional_feasibility(inventory: Inventory, *, seed: int = 42) -> dict[str, Any]:
    """Assess the illustrative table, not a G2 allocation; development is additional."""
    groups = _eligible_groups(inventory)
    requests = {role: {"count": count} for role, count in PROVISIONAL_COUNTS.items()}
    report = {
        "schema_version": 1,
        "inventory_sha256": inventory.manifest["inventory_sha256"],
        "provisional_total": 1500,
        "provisional_counts": PROVISIONAL_COUNTS,
        "frozen": False,
        "class_counts_specified": False,
        "development": {"required": True, "count": None, "included_in_1500": False},
        "video_holdout": False,
        "seed": seed,
    }
    try:
        assigned = _partition(groups, requests, seed)
    except AllocationInfeasible as error:
        report.update(status="infeasible", infeasibility=error.report)
    else:
        used = {group for values in assigned.values() for group in values}
        report.update(
            status="provisional_counts_feasible_not_frozen",
            remaining_eligible_examples=sum(
                len(rows) for group, rows in groups.items() if group not in used
            ),
            remaining_groups=len(groups) - len(used),
        )
    report["qualification_boundary"] = (
        "No development budget, class/prior-shift requirements, tuning precision, "
        "window feasibility, or scientific qualification established. "
        "No role membership frozen by this report."
    )
    return _seal(report, "report_sha256")


def _validate_allocation(
    inventory: Inventory, allocation: Mapping[str, Any]
) -> dict[str, list[Any]]:
    payload = {key: value for key, value in allocation.items() if key != "allocation_sha256"}
    if (
        allocation.get("allocation_sha256") != sha256_json(payload)
        or allocation.get("inventory_sha256") != inventory.manifest["inventory_sha256"]
    ):
        raise ValueError("allocation identity/hash does not match inventory")
    groups = _eligible_groups(inventory)
    assigned = set()
    if set(allocation["roles"]) != set(REQUIRED_ROLES):
        raise ValueError("allocation does not contain all mandatory roles")
    for role, data in allocation["roles"].items():
        ids = data["group_ids"]
        if (
            len(ids) != len(set(ids))
            or assigned.intersection(ids)
            or any(group not in groups for group in ids)
        ):
            raise ValueError(f"invalid/overlapping group membership in {role}")
        sources = sorted(row.source_id for group in ids for row in groups[group])
        if data["source_ids"] != sources or data["count"] != len(sources):
            raise ValueError(f"split group or invalid membership in {role}")
        assigned.update(ids)
    return groups


def build_schedule(
    inventory: Inventory,
    allocation: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    schedule_id: str,
    paired_with: Mapping[str, Any] | None = None,
    reuse_declaration: Mapping[str, Any] | None = None,
    variants: Mapping[str, Mapping[str, Any]] | None = None,
    transformation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Place fixed-role whole groups exactly once; paired controls copy placement.

    `variants` maps source IDs to explicit text and/or binary label overrides;
    transformed inputs stay in their source role/group. Their exact values and
    declared transformation are hashed, never silently normalized.
    """
    if not isinstance(schedule_id, str) or not schedule_id:
        raise ValueError("schedule_id must be nonempty")
    groups = _validate_allocation(inventory, allocation)
    variants = dict(variants or {})
    if variants and not transformation:
        raise ValueError("variants require a documented transformation")
    if paired_with is not None and not reuse_declaration:
        raise ValueError("paired controls require an explicit reuse/dependence declaration")
    requests = {}
    by_role: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for index, window in enumerate(windows):
        role = window.get("role")
        if role not in allocation["roles"]:
            raise ValueError(f"window {index} has unknown role {role}")
        normalized = _requests({str(index): window})[str(index)]
        requests[str(index)] = normalized
        by_role[role][str(index)] = normalized
    if paired_with is not None:
        paired_payload = {
            key: value for key, value in paired_with.items() if key != "schedule_sha256"
        }
        if (
            paired_with.get("schedule_sha256") != sha256_json(paired_payload)
            or paired_with.get("allocation_sha256") != allocation["allocation_sha256"]
        ):
            raise ValueError("paired schedule hash/allocation mismatch")
        if len(paired_with["windows"]) != len(windows):
            raise ValueError("paired controls must use the same window count")
        assigned = {}
        for index, (window, original) in enumerate(zip(windows, paired_with["windows"])):
            if (
                original["role"] != window["role"]
                or original["count"] != requests[str(index)]["count"]
            ):
                raise ValueError("paired control changed role/window membership request")
            assigned[str(index)] = original["group_ids"]
    else:
        assigned = {}
        for role, role_requests in by_role.items():
            pool = {group: groups[group] for group in allocation["roles"][role]["group_ids"]}
            role_seed = int(sha256_json([seed, role]), 16)
            assigned.update(_partition(pool, role_requests, role_seed))
    output = []
    seen = set()
    all_sources = set()
    rng = random.Random(seed)
    for index, window in enumerate(windows):
        group_ids = sorted(assigned[str(index)])
        if (
            len(group_ids) != len(set(group_ids))
            or seen.intersection(group_ids)
            or not set(group_ids).issubset(allocation["roles"][window["role"]]["group_ids"])
        ):
            raise ValueError("duplicate source group across windows or role mismatch")
        seen.update(group_ids)
        rows = sorted(
            (row for group in group_ids for row in groups[group]), key=lambda row: row.source_id
        )
        classes = dict(Counter(str(row.label) for row in rows))
        requested_classes = requests[str(index)]["class_counts"]
        if len(rows) != requests[str(index)]["count"] or (
            requested_classes is not None
            and any(classes.get(key, 0) != value for key, value in requested_classes.items())
        ):
            raise ValueError(f"paired membership violates window {index} source-class counts")
        if paired_with is not None:
            if sorted(paired_with["windows"][index]["source_ids"]) != sorted(
                row.source_id for row in rows
            ):
                raise ValueError("paired control does not retain exact source membership")
            order = paired_with["windows"][index]["source_ids"]
        else:
            order = [row.source_id for row in rows]
            rng.shuffle(order)
        all_sources.update(order)
        row_by_id = {row.source_id: row for row in rows}
        samples = []
        for source_id in order:
            row = row_by_id[source_id]
            variant = variants.get(source_id, {})
            if set(variant) - {"text", "label"}:
                raise ValueError("variants accept only explicit text and label")
            text = variant.get("text", row.text)
            label = variant.get("label", row.label)
            if not isinstance(text, str) or type(label) is not int or label not in (0, 1):
                raise ValueError("variant must retain string text and binary integer label")
            samples.append(
                {
                    "source_id": source_id,
                    "group_id": row.group_id,
                    "video": row.video,
                    "text": text,
                    "label": label,
                    "source_text_sha256": sha256_json(row.text),
                    "text_sha256": sha256_json(text),
                    "variant": source_id in variants,
                }
            )
        output.append(
            {
                "window_index": index,
                "role": window["role"],
                "count": len(rows),
                "group_ids": group_ids,
                "source_ids": order,
                "source_class_counts": classes,
                "class_counts": dict(Counter(str(sample["label"]) for sample in samples)),
                "samples": samples,
            }
        )
    if set(variants) - all_sources:
        raise ValueError("variant source IDs are outside this schedule")
    reuse = dict(
        reuse_declaration
        or {
            "within_schedule": "none",
            "across_schedules": (
                "resampling within fixed role banks; shared-source horizons are "
                "dependent, not independent binomial trials"
            ),
        }
    )
    return _seal(
        {
            "schema_version": 1,
            "schedule_id": schedule_id,
            "seed": seed,
            "inventory_sha256": inventory.manifest["inventory_sha256"],
            "allocation_sha256": allocation["allocation_sha256"],
            "task_schema_sha256": inventory.manifest["hashes"]["task_schema"],
            "source_transforms_sha256": inventory.manifest["hashes"]["transforms"],
            "window_requests": [
                {"role": window["role"], **requests[str(index)]}
                for index, window in enumerate(windows)
            ],
            "windows": output,
            "membership_sha256": sha256_json(
                [[window["role"], window["source_ids"], window["group_ids"]] for window in output]
            ),
            "text_sha256": sha256_json(
                [
                    [sample["source_id"], sample["text"]]
                    for window in output
                    for sample in window["samples"]
                ]
            ),
            "labels_sha256": sha256_json(
                [
                    [sample["source_id"], sample["label"]]
                    for window in output
                    for sample in window["samples"]
                ]
            ),
            "transformation": dict(transformation or {"input": "identity"}),
            "transformation_sha256": sha256_json(transformation or {"input": "identity"}),
            "reuse": reuse,
            "paired_schedule_sha256": None
            if paired_with is None
            else paired_with["schedule_sha256"],
        },
        "schedule_sha256",
    )
