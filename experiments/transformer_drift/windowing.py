"""Partition AG News test set into temporal windows with drift injection."""

from __future__ import annotations

import numpy as np
from datasets import Dataset, load_dataset

from .config import DATASET_NAME, DRIFT_START_WINDOW, N_WINDOWS, WINDOW_SIZE
from .drift_injection import DRIFT_INJECTORS


def load_ag_news_test() -> Dataset:
    """Load the AG News test split."""
    return load_dataset(DATASET_NAME, split="test")


def create_windows(
    dataset: Dataset,
    n_windows: int = N_WINDOWS,
    window_size: int = WINDOW_SIZE,
    seed: int = 42,
) -> list[Dataset]:
    """Partition dataset into fixed-size windows via random sampling.

    Each window is an independent random sample (with replacement across
    windows) to simulate a temporal stream of incoming data.

    Args:
        dataset: Full dataset to sample from.
        n_windows: Number of windows to create.
        window_size: Samples per window.
        seed: Random seed for reproducibility.

    Returns:
        List of Dataset objects, one per window.
    """
    rng = np.random.RandomState(seed)
    windows = []
    for _ in range(n_windows):
        indices = rng.choice(len(dataset), size=window_size, replace=False)
        window = dataset.select(indices.tolist())
        windows.append(window)
    return windows


def create_drifted_windows(
    drift_type: str,
    n_windows: int = N_WINDOWS,
    window_size: int = WINDOW_SIZE,
    start_window: int = DRIFT_START_WINDOW,
    seed: int = 42,
) -> list[Dataset]:
    """Create windows with synthetic drift injection.

    Windows before ``start_window`` are clean (baseline/calibration).
    Windows from ``start_window`` onward have increasing drift applied.

    Args:
        drift_type: One of "domain_shift", "vocabulary_shift",
            "class_distribution_shift".
        n_windows: Total number of windows.
        window_size: Samples per window.
        start_window: Window index where drift begins.
        seed: Random seed.

    Returns:
        List of Dataset objects with drift applied to later windows.
    """
    injector = DRIFT_INJECTORS[drift_type]
    test_data = load_ag_news_test()

    # Create base windows (clean random samples)
    base_windows = create_windows(test_data, n_windows, window_size, seed)

    # Apply drift injection to each window
    drifted_windows = []
    for wid, window in enumerate(base_windows):
        drifted = injector(
            window,
            window_id=wid,
            start_window=start_window,
            n_windows=n_windows,
            seed=seed,
        )
        drifted_windows.append(drifted)

    return drifted_windows
