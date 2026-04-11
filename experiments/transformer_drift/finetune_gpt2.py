"""Fine-tune GPT-2 small on AG News for 4-class classification.

Full fine-tune (no LoRA) — GPT-2 small is only ~124M params and fits
easily on an RTX 3070.  Saves the model + tokenizer to
config.MODELS_DIR / "gpt2-agnews".

Usage:
    PYTHONPATH=/home/brendenadm/projects python experiments/transformer_drift/finetune_gpt2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import torch
from datasets import load_dataset
from transformers import (
    DataCollatorWithPadding,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

from experiments.transformer_drift.config import (
    DATASET_NAME,
    EVAL_BATCH_SIZE,
    GPT2_MODEL_NAME,
    MAX_SEQ_LENGTH,
    MODELS_DIR,
    NUM_EPOCHS,
    NUM_LABELS,
    TRAIN_BATCH_SIZE,
    WARMUP_RATIO,
)


def main():
    output_dir = MODELS_DIR / "gpt2-agnews"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer — GPT-2 has no pad token by default
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = GPT2ForSequenceClassification.from_pretrained(
        GPT2_MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load and tokenize AG News
    dataset = load_dataset(DATASET_NAME)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training — full fine-tune (no LoRA), small enough to fit
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=5e-5,  # standard for full fine-tune
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    trainer.train()

    # Save final model
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")

    # Quick evaluation
    results = trainer.evaluate()
    print(f"Eval loss: {results['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
