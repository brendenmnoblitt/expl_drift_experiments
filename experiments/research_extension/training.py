"""Train local BERT/GPT-2 task classifiers from pinned, group-disjoint roles.

Only training updates weights; only development selects an epoch. This command
never scores reference, tuning or held-out roles and does not establish G1.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from .data import Inventory, save_artifact, sha256_json
from .reference_adapter import checkpoint_fingerprint
from .schedule import _validate_allocation


def prepare_training_data(inventory: Inventory, allocation: dict[str, Any]) -> dict[str, Any]:
    """Validate complete role membership before loading weights or doing inference."""
    _validate_allocation(inventory, allocation)
    schema = inventory.manifest["task_schema"]
    candidates = schema["candidate_ids"]
    if (
        len(candidates) < 2
        or len(set(candidates)) != len(candidates)
        or any(not isinstance(candidate, str) or not candidate for candidate in candidates)
        or schema["labels"] != {str(i): name for i, name in enumerate(candidates)}
        or schema["reference_candidate"] not in candidates
    ):
        raise ValueError("task schema requires dense, ordered classifier labels")
    records = {row.source_id: row for row in inventory.records}
    if len(records) != len(inventory.records):
        raise ValueError("source IDs must be unique")
    result: dict[str, Any] = {"task_schema": schema}
    for role in ("training", "development"):
        selected = [records[source_id] for source_id in allocation["roles"][role]["source_ids"]]
        if not selected:
            raise ValueError(f"{role} must contain designated examples")
        if any(
            row.exclusions or type(row.label) is not int or not 0 <= row.label < len(candidates)
            for row in selected
        ):
            raise ValueError(f"{role} contains excluded or incompatible labels")
        result[role] = selected
    return result


def _development_metrics(model: Any, encoded: list[dict], collator: Any, batch_size: int) -> dict:
    import torch

    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch = collator(encoded[start : start + batch_size]).to(model.device)
            labels.append(batch.pop("labels").cpu())
            logits.append(model(**batch).logits.float().cpu())
    scores = torch.cat(logits)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("development logits are non-finite; incomplete run retained")
    truth = torch.cat(labels)
    prediction = scores.argmax(dim=-1)
    probabilities = scores.softmax(dim=-1)
    recalls = {}
    for label in range(model.config.num_labels):
        mask = truth == label
        recalls[str(label)] = (
            float((prediction[mask] == label).float().mean()) if mask.any() else None
        )
    defined = [value for value in recalls.values() if value is not None]
    return {
        "count": len(truth),
        "accuracy": float((prediction == truth).float().mean()),
        "balanced_accuracy": sum(defined) / len(defined),
        "per_class_recall": recalls,
        "log_loss": float(torch.nn.functional.cross_entropy(scores, truth)),
        "multiclass_brier": float(
            (probabilities - torch.nn.functional.one_hot(truth, model.config.num_labels))
            .square()
            .sum(dim=-1)
            .mean()
        ),
    }


def train_reference(
    inventory: Inventory,
    allocation: dict[str, Any],
    base_checkpoint: str | Path,
    output: str | Path,
    *,
    expected_base_sha256: str,
    epochs: int = 3,
    batch_size: int = 8,
    max_tokens: int = 128,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Full float32 fine-tuning; fixed epoch budget and minimum development loss selection."""
    data = prepare_training_data(inventory, allocation)
    for name, value, minimum in (
        ("epochs", epochs, 1),
        ("batch_size", batch_size, 1),
        ("max_tokens", max_tokens, 2),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            "training requires a fresh output directory; partial runs are not resumed"
        )
    base_checkpoint = Path(base_checkpoint)
    actual_base_sha256 = checkpoint_fingerprint(base_checkpoint)
    if actual_base_sha256 != expected_base_sha256:
        raise ValueError("base checkpoint content/hash mismatch")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )

    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    schema = data["task_schema"]
    candidates = schema["candidate_ids"]
    config = AutoConfig.from_pretrained(base_checkpoint, local_files_only=True)
    if config.model_type not in ("bert", "gpt2"):
        raise ValueError("only BERT and GPT-2 reference architectures are supported")
    if len(candidates) not in (2, 4):
        raise ValueError("study reference tasks require two or four candidates")
    positions = getattr(config, "max_position_embeddings", getattr(config, "n_positions", None))
    if positions is not None and max_tokens > positions:
        raise ValueError("max_tokens exceeds checkpoint positional capacity")
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if config.model_type != "gpt2" or tokenizer.eos_token_id is None:
            raise ValueError("checkpoint requires a pad token or GPT-2 EOS identity")
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    config.num_labels = len(candidates)
    config.id2label = dict(enumerate(candidates))
    config.label2id = {name: i for i, name in enumerate(candidates)}
    if config.model_type == "gpt2":
        config.use_cache = False
    plan = {
        "schema_version": 1,
        "inventory_sha256": inventory.manifest["inventory_sha256"],
        "allocation_sha256": allocation["allocation_sha256"],
        "task_schema": schema,
        "base_checkpoint_sha256": actual_base_sha256,
        "base_checkpoint": str(base_checkpoint.resolve()),
        "model_type": config.model_type,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "training_method": "full_fine_tune",
        "precision": "float32_no_autocast",
        "tf32": False,
        "deterministic_algorithms": True,
        "attention_implementation": "eager",
        "padding_side": "right",
        "pad_token_id": tokenizer.pad_token_id,
        "selection": "minimum_development_log_loss; earliest epoch on ties",
        "source_ids": {
            role: [row.source_id for row in data[role]] for role in ("training", "development")
        },
        "group_ids": {
            role: sorted({row.group_id for row in data[role]})
            for role in ("training", "development")
        },
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "software": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "tokenizers")
        },
        "device": device,
        "cuda_device": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "G1": "not_established: training provenance is not scientific extraction qualification",
    }
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    # Atomically reserve the output before any model forward or weight update.
    output.mkdir(parents=True, exist_ok=False)
    save_artifact(output / "training-plan.json", plan)
    started = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        base_checkpoint,
        config=config,
        local_files_only=True,
        ignore_mismatched_sizes=True,
        dtype="float32",
        attn_implementation="eager",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    encoded = {}
    for role in ("training", "development"):
        tokens = tokenizer(
            [row.text for row in data[role]], truncation=True, max_length=max_tokens, padding=False
        )
        encoded[role] = [
            {**{key: values[i] for key, values in tokens.items()}, "labels": row.label}
            for i, row in enumerate(data[role])
        ]
    initial = _development_metrics(model, encoded["development"], collator, batch_size)
    epoch_reports = []
    best_loss = math.inf
    best_epoch = 0
    steps = 0
    order = list(range(len(encoded["training"])))
    generator = random.Random(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        generator.shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = collator([encoded["training"][i] for i in indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss became non-finite; incomplete run retained")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            steps += 1
        development = _development_metrics(model, encoded["development"], collator, batch_size)
        epoch_reports.append(
            {
                "epoch": epoch,
                "training_log_loss": total_loss / len(order),
                "development": development,
            }
        )
        if development["log_loss"] < best_loss:
            best_loss, best_epoch = development["log_loss"], epoch
            model.save_pretrained(output / "checkpoint")
            tokenizer.save_pretrained(output / "checkpoint")
    report = {
        "schema_version": 1,
        "training_plan_sha256": sha256_json(plan),
        "checkpoint_sha256": checkpoint_fingerprint(output / "checkpoint"),
        "selected_epoch": best_epoch,
        "initial_development": initial,
        "epochs": epoch_reports,
        "optimizer_steps": steps,
        "training_examples": len(encoded["training"]),
        "development_examples": len(encoded["development"]),
        "held_out_model_evaluations": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None
        ),
        "G1": plan["G1"],
    }
    save_artifact(output / "training.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = train_reference(
        Inventory.from_artifact(json.loads(args.inventory.read_text(encoding="utf-8"))),
        json.loads(args.allocation.read_text(encoding="utf-8")),
        args.base_checkpoint,
        args.output,
        expected_base_sha256=args.base_sha256,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
