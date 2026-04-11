"""Shared configuration for Transformer drift detection experiments."""

from pathlib import Path

# === Paths ===
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
MODELS_DIR = EXPERIMENT_DIR / "models"

# === Dataset ===
DATASET_NAME = "ag_news"
NUM_LABELS = 4
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
MAX_SEQ_LENGTH = 128

# === Models ===
BERT_MODEL_NAME = "google-bert/bert-base-uncased"
GPT2_MODEL_NAME = "openai-community/gpt2"

# === Fine-tuning ===
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
WARMUP_RATIO = 0.1

# === Drift experiment ===
SEEDS = list(range(10))
N_WINDOWS = 20
N_CALIBRATION = 4
WINDOW_SIZE = 500  # samples per window
WARNING_STD = 2.5
CRITICAL_STD = 3.5
DRIFT_START_WINDOW = 5  # window index where drift injection begins

# === Attribution extraction ===
ATTENTION_STRATEGY = "last_layer_mean"
ATTENTION_BATCH_SIZE = 32
IG_N_STEPS = 50
IG_BATCH_SIZE = 2  # must be small for IG gradient memory on 8GB VRAM

# === Drift scenarios ===
DRIFT_TYPES = ["domain_shift", "vocabulary_shift", "class_distribution_shift"]
ATTRIBUTION_METHODS = ["attention", "integrated_gradients"]
MODEL_TYPES = ["bert", "gpt2"]
