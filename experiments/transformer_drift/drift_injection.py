"""Synthetic drift injection for AG News text classification.

Three drift scenarios applied to a frozen model's input stream:
1. Domain shift — gradually replace one category with another
2. Vocabulary shift — inject out-of-domain tokens into texts
3. Class distribution shift — alter category proportions over time
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset


def inject_domain_shift(
    dataset: Dataset,
    window_id: int,
    start_window: int,
    n_windows: int = 20,
    source_label: int = 2,  # Business
    target_label: int = 1,  # Sports
    seed: int = 42,
) -> Dataset:
    """Gradually replace source-label articles with target-label articles.

    After ``start_window``, an increasing fraction of source-label
    articles are swapped with randomly sampled target-label articles.
    The fraction reaches 1.0 at the final window.

    Args:
        dataset: HuggingFace Dataset with "text" and "label" columns.
        window_id: Current window index.
        start_window: Window to begin injection.
        n_windows: Total windows in experiment.
        source_label: Category to replace (default: 2 = Business).
        target_label: Category to sample replacements from (default: 1 = Sports).
        seed: Random seed.

    Returns:
        Modified dataset with domain shift applied.
    """
    if window_id < start_window:
        return dataset

    rng = np.random.RandomState(seed + window_id)
    # Cap at 0.5 so late windows still have some correctly-labeled source
    # samples — this keeps accuracy from cratering too fast for the
    # attribution signal to lead.
    fraction = min(0.65, 0.65 * (window_id - start_window) / max(1, n_windows - start_window))

    labels = np.array(dataset["label"])
    texts = list(dataset["text"])

    source_indices = np.where(labels == source_label)[0]
    target_indices = np.where(labels == target_label)[0]

    n_to_replace = int(fraction * len(source_indices))
    if n_to_replace == 0 or len(target_indices) == 0:
        return dataset

    replace_indices = rng.choice(source_indices, size=n_to_replace, replace=False)
    donor_indices = rng.choice(target_indices, size=n_to_replace, replace=True)

    # Swap text but keep the ORIGINAL source_label — this creates a real
    # text/label mismatch the classifier has to fail on.  (Previously we
    # relabeled to target_label, which made every swapped sample still
    # correctly labeled AG News data, so accuracy never dropped.)
    donor_texts = list(dataset["text"])
    for src_idx, donor_idx in zip(replace_indices, donor_indices):
        texts[src_idx] = donor_texts[donor_idx]

    return Dataset.from_dict({"text": texts, "label": labels.tolist()})


def inject_vocabulary_shift(
    dataset: Dataset,
    window_id: int,
    start_window: int,
    n_windows: int = 20,
    seed: int = 42,
) -> Dataset:
    """Inject out-of-domain tokens into article texts.

    After ``start_window``, a growing fraction of tokens in each text
    are replaced with domain-irrelevant tokens (e.g., technical jargon,
    random strings).  This simulates vocabulary drift without changing
    labels or class proportions.

    Args:
        dataset: HuggingFace Dataset with "text" and "label" columns.
        window_id: Current window index.
        start_window: Window to begin injection.
        n_windows: Total windows in experiment.
        seed: Random seed.

    Returns:
        Modified dataset with vocabulary shift applied.
    """
    if window_id < start_window:
        return dataset

    rng = np.random.RandomState(seed + window_id)
    fraction = min(0.3, 0.3 * (window_id - start_window) / max(1, n_windows - start_window))

    # Label-neutral pseudo-word tokens.  We avoid the previous list
    # (blockchain / kubernetes / algorithm / ...) because those are strongly
    # Sci/Tech-coded in AG News, so injecting them flipped labels instead
    # of adding neutral noise — and accuracy then collapsed faster than
    # attention patterns could drift.
    ood_tokens = [
        "flendor", "gravnit", "blurpex", "zemnic", "qalber", "frimpton",
        "xorbal", "wazzit", "gromlik", "oszkar", "plovith", "brantis",
        "yurlock", "krenbal", "mivolon", "tovrith", "axentil", "drelvan",
        "sprogil", "vuntrex",
    ]

    texts = list(dataset["text"])
    for i in range(len(texts)):
        words = texts[i].split()
        n_inject = max(1, int(fraction * len(words)))
        inject_positions = rng.choice(len(words), size=min(n_inject, len(words)), replace=False)
        for pos in inject_positions:
            words[pos] = rng.choice(ood_tokens)
        texts[i] = " ".join(words)

    return Dataset.from_dict({"text": texts, "label": dataset["label"]})


def inject_class_distribution_shift(
    dataset: Dataset,
    window_id: int,
    start_window: int,
    n_windows: int = 20,
    dominant_label: int = 0,  # World
    seed: int = 42,
) -> Dataset:
    """Shift class proportions toward a dominant category.

    After ``start_window``, the proportion of ``dominant_label`` samples
    gradually increases while other classes are undersampled.  At full
    drift, the dominant class comprises ~70% of the window.

    Args:
        dataset: HuggingFace Dataset with "text" and "label" columns.
        window_id: Current window index.
        start_window: Window to begin injection.
        n_windows: Total windows in experiment.
        dominant_label: Category to over-represent (default: 0 = World).
        seed: Random seed.

    Returns:
        Modified dataset with shifted class distribution.
    """
    if window_id < start_window:
        return dataset

    rng = np.random.RandomState(seed + window_id)
    # Target: dominant class goes from 25% (balanced) to 70%
    target_dominant_frac = 0.25 + 0.45 * min(
        1.0, (window_id - start_window) / max(1, n_windows - start_window)
    )

    labels = np.array(dataset["label"])
    n_total = len(labels)

    dominant_indices = np.where(labels == dominant_label)[0]
    other_indices = np.where(labels != dominant_label)[0]

    n_dominant = int(target_dominant_frac * n_total)
    n_other = n_total - n_dominant

    # Resample with replacement to hit target proportions
    sampled_dominant = rng.choice(dominant_indices, size=n_dominant, replace=True)
    sampled_other = rng.choice(other_indices, size=n_other, replace=True)

    all_indices = np.concatenate([sampled_dominant, sampled_other])
    rng.shuffle(all_indices)

    texts = [dataset["text"][i] for i in all_indices]
    new_labels = [dataset["label"][i] for i in all_indices]

    # Without this, pure resampling preserves all (text, label) correctness
    # and accuracy never drops.  Swap a growing fraction of dominant-class
    # samples' text with an OTHER-class donor text, keeping the dominant
    # label — so those samples become mislabeled and accuracy falls.
    # Cap at 0.5 so late windows don't crater accuracy faster than the
    # attribution signal can ramp.
    corrupt_frac = min(
        0.5, 0.5 * (window_id - start_window) / max(1, n_windows - start_window)
    )
    dominant_positions = [i for i, lab in enumerate(new_labels) if lab == dominant_label]
    n_corrupt = int(corrupt_frac * len(dominant_positions))
    if n_corrupt > 0 and len(other_indices) > 0:
        positions_to_corrupt = rng.choice(dominant_positions, size=n_corrupt, replace=False)
        donor_indices = rng.choice(other_indices, size=n_corrupt, replace=True)
        for pos, donor in zip(positions_to_corrupt, donor_indices):
            texts[pos] = dataset["text"][donor]

    return Dataset.from_dict({"text": texts, "label": new_labels})


def inject_preamble_shift(
    dataset: Dataset,
    window_id: int,
    start_window: int,
    n_windows: int = 20,
    seed: int = 42,
    max_preamble_tokens: int = 120,
) -> Dataset:
    """Prepend a growing run of neutral stop-word filler to each article.

    Designed to induce a gradual attention-pattern shift without immediately
    breaking the task: early windows add a handful of filler tokens (content
    still intact, attention mass moves slightly to earlier positions); late
    windows push content past MAX_SEQ_LENGTH=128 and accuracy degrades.

    Uses common English stop words as filler because they carry minimal
    class signal and BERT tokenizes them to single pieces.

    Args:
        dataset: HuggingFace Dataset with "text" and "label" columns.
        window_id: Current window index.
        start_window: Window to begin injection.
        n_windows: Total windows in experiment.
        seed: Random seed.
        max_preamble_tokens: Preamble length at final window.

    Returns:
        Modified dataset with preamble prepended.
    """
    if window_id < start_window:
        return dataset

    rng = np.random.RandomState(seed + window_id)
    fraction = min(1.0, (window_id - start_window) / max(1, n_windows - start_window))
    n_preamble = int(fraction * max_preamble_tokens)
    if n_preamble == 0:
        return dataset

    stop_words = [
        "the", "a", "of", "and", "to", "in", "is", "it", "that", "was",
        "be", "but", "for", "on", "by", "with", "as", "at", "from", "or",
    ]
    preamble_tokens = rng.choice(stop_words, size=n_preamble, replace=True)
    preamble = " ".join(preamble_tokens)

    texts = [f"{preamble} {t}" for t in dataset["text"]]
    return Dataset.from_dict({"text": texts, "label": list(dataset["label"])})


# Registry for easy parameterization
DRIFT_INJECTORS = {
    "domain_shift": inject_domain_shift,
    "vocabulary_shift": inject_vocabulary_shift,
    "class_distribution_shift": inject_class_distribution_shift,
    "preamble_shift": inject_preamble_shift,
}
