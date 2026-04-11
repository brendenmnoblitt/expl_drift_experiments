"""Fine-tune Phi-2 on AG News for 4-class classification.

Uses QLoRA (4-bit quantization via bitsandbytes) for memory-efficient
training on an RTX 3070 (8GB VRAM).  Saves the merged model + tokenizer
to config.MODELS_DIR / "phi2-agnews".

Usage:
    PYTHONPATH=/home/brendenadm/projects python experiments/transformer_drift/finetune_phi2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from experiments.transformer_drift.config import (
    DATASET_NAME,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
    MODELS_DIR,
    NUM_EPOCHS,
    NUM_LABELS,
    PHI2_MODEL_NAME,
    TRAIN_BATCH_SIZE,
    WARMUP_RATIO,
)


def main():
    output_dir = MODELS_DIR / "phi2-agnews"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer — Phi-2 doesn't have a pad token by default
    tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    model = AutoModelForSequenceClassification.from_pretrained(
        PHI2_MODEL_NAME,
        num_labels=NUM_LABELS,
        quantization_config=bnb_config,
        device_map="auto",
    )
    # Use pad token id for the classification head
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA — target Phi-2's attention projections
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
    # Subsample training set for speed — 30K is plenty for 4-class with 2.7B params
    tokenized["train"] = tokenized["train"].shuffle(seed=42).select(range(30000))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training — 1 epoch with larger effective batch for speed
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=4,  # limited by 8GB VRAM
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # effective batch size 8
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        fp16=True,
        report_to="none",
        seed=42,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    trainer.train()

    # Merge LoRA weights and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")

    # Quick evaluation
    results = trainer.evaluate()
    print(f"Eval loss: {results['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
