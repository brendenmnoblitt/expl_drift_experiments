"""Synthetic drift injection functions for experiment windows."""

import numpy as np
import pandas as pd


def inject_covariate_drift(
    X: pd.DataFrame,
    window_id: int,
    feature: str,
    start_window: int,
    severity: float = 1.0,
    seed: int | None = None,
    mode: str = "shift",
) -> pd.DataFrame:
    """Apply covariate drift to a feature column starting at start_window.

    Args:
        X: Input DataFrame (unchanged if window_id < start_window).
        window_id: Current window index.
        feature: Column name to apply drift to.
        start_window: Window index to start applying drift.
        severity: Controls the magnitude of drift (default 1.0).
        seed: Random seed for noise generation (only used if mode="noise").
        mode: Type of drift to apply:
            - "shift": Add a linearly increasing offset to the feature.
            - "noise": Add Gaussian noise with increasing standard deviation.
    Returns:
        pd.DataFrame: Modified copy of X with drift applied to the specified feature.
    """
    X_out = X.copy()
    if window_id >= start_window:
        offset = window_id - start_window
        if mode == "shift":
            X_out[feature] = X_out[feature] + severity * offset
        elif mode == "noise":
            rng = np.random.RandomState(seed if seed is not None else 0 + window_id)
            scale = severity * offset
            noise = rng.normal(0, max(scale, 1e-10), size=len(X_out))
            X_out[feature] = X_out[feature] + noise
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'shift' or 'noise'.")
    return X_out


def inject_concept_drift(
    X: pd.DataFrame,
    y: pd.Series,
    window_id: int,
    original_feature: str,
    new_feature: str,
    start_window: int,
    severity: float = 1.0,
    n_windows: int = 20,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Re-label a fraction of samples based on a different feature.

    After start_window, an increasing fraction of labels are reassigned based on
    a median-split of new_feature instead of the original labels. The fraction grows linearly
    with window_id, reaching severity at the end of the experiment
    (window_id = start_window + n_windows). X is returned unchanged.

    Args:
        X: Input DataFrame (unchanged).
        y: Original labels (pd.Series).
        window_id: Current window index.
        original_feature: Feature originally used for labeling (not used in this function but
            included for consistency with other drift functions).
        new_feature: Feature to use for re-labeling after drift starts.
        start_window: Window index to start applying drift.
        severity: Maximum fraction of labels to reassign at the end of the experiment (default
            1.0).
        n_windows: Number of windows over which to increase the fraction of affected samples.
        seed: Random seed for selecting which samples to re-label.
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X unchanged, y with concept drift applied)
    """
    y_out = y.copy()
    if window_id >= start_window:
        rng = np.random.RandomState(seed if seed is not None else 0 + window_id)
        fraction = min(1.0, severity * (window_id - start_window) / n_windows)
        n_to_flip = int(fraction * len(y_out))
        if n_to_flip > 0:
            flip_indices = rng.choice(len(y_out), size=n_to_flip, replace=False)
            median_val = X[new_feature].median()
            new_labels = (X[new_feature].iloc[flip_indices] > median_val).astype(int)
            y_out.iloc[flip_indices] = new_labels.values
    return X, y_out


def inject_concept_drift_rotation(
    X: pd.DataFrame,
    y: pd.Series,
    window_id: int,
    original_feature: str,
    new_feature: str,
    start_window: int,
    severity: float = 1.0,
    n_windows: int = 20,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Rotate the decision boundary from original_feature to new_feature.

    A growing fraction of samples have their labels replaced by a threshold
    on a weighted combination:
        score = (1 - alpha) * original_feature + alpha * new_feature
    where alpha = min(1.0, severity * (window_id - start_window) / n_windows).

    The fraction of affected samples grows at the same rate as alpha, ensuring
    gradual degradation comparable to :func:`inject_concept_drift`.
    X is returned unchanged.

    Args:
        X: Input DataFrame (unchanged).
        y: Original labels (pd.Series).
        window_id: Current window index.
        original_feature: Feature originally used for labeling (used in the weighted combination).
        new_feature: Feature to rotate towards (used in the weighted combination).
        start_window: Window index to start applying drift.
        severity: Controls how quickly the decision boundary rotates (default 1.0).
        n_windows: Number of windows over which to complete the rotation.
        seed: Random seed for selecting which samples to re-label.
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X unchanged, y with concept drift applied)
    """
    y_out = y.copy()
    if window_id >= start_window:
        rng = np.random.RandomState(seed if seed is not None else 0 + window_id)
        alpha = min(1.0, severity * (window_id - start_window) / n_windows)
        fraction = alpha  # grow affected population at same rate as boundary shift
        n_to_flip = int(fraction * len(y_out))
        if n_to_flip > 0:
            score = (1 - alpha) * X[original_feature] + alpha * X[new_feature]
            new_labels = (score > score.median()).astype(int)
            flip_indices = rng.choice(len(y_out), size=n_to_flip, replace=False)
            y_out.iloc[flip_indices] = new_labels.iloc[flip_indices]
    return X, y_out


def inject_concept_drift_perturbation(
    X: pd.DataFrame,
    y: pd.Series,
    window_id: int,
    features: list[str],
    start_window: int,
    severity: float = 1.0,
    n_windows: int = 20,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Perturb the labeling function coefficients with growing noise.

    A growing fraction of samples have their labels replaced by a threshold
    on a linear combination of ``features`` with noisy coefficients:
    coefficients = 1.0 + N(0, sigma) where
    sigma = severity * (window_id - start_window) / n_windows.

    The fraction of affected samples equals sigma (capped at 1.0), ensuring
    gradual degradation comparable to :func:`inject_concept_drift`.
    X is returned unchanged.

    Args:
        X: Input DataFrame (unchanged).
        y: Original labels (pd.Series).
        window_id: Current window index.
        features: List of feature names to use in the linear combination for re-labeling.
        start_window: Window index to start applying drift.
        severity: Controls the magnitude of coefficient perturbation (default 1.0).
        n_windows: Number of windows over which to increase the fraction of affected samples.
        seed: Random seed for selecting which samples to re-label and for noise generation.
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X unchanged, y with concept drift applied)
    """
    y_out = y.copy()
    if window_id >= start_window:
        rng = np.random.RandomState(seed if seed is not None else 0 + window_id)
        sigma = severity * (window_id - start_window) / n_windows
        fraction = min(1.0, sigma)
        n_to_flip = int(fraction * len(y_out))
        if n_to_flip > 0:
            perturbation = rng.normal(0, max(sigma, 1e-10), size=len(features))
            coefficients = 1.0 + perturbation
            score = X[features].values @ coefficients
            new_labels = pd.Series(
                (score > np.median(score)).astype(int), index=y.index
            )
            flip_indices = rng.choice(len(y_out), size=n_to_flip, replace=False)
            y_out.iloc[flip_indices] = new_labels.iloc[flip_indices]
    return X, y_out


def inject_concept_drift_conditional(
    X: pd.DataFrame,
    y: pd.Series,
    window_id: int,
    original_feature: str,
    new_feature: str,
    condition_feature: str,
    start_window: int,
    severity: float = 1.0,
    n_windows: int = 20,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Change the feature-label relationship only for a subpopulation.

    For samples where condition_feature > median, a growing fraction have
    their labels replaced by a rotation from original_feature to new_feature
    (same mechanism as :func:`inject_concept_drift_rotation`). Samples at or
    below the median keep their original labels. X is returned unchanged.
    
    Args:
        X: Input DataFrame (unchanged).
        y: Original labels (pd.Series).
        window_id: Current window index.
        original_feature: Feature originally used for labeling (used in the weighted combination).
        new_feature: Feature to rotate towards (used in the weighted combination).
        condition_feature: Feature used to select the subpopulation affected by drift.
        start_window: Window index to start applying drift.
        severity: Controls how quickly the decision boundary rotates for the affected subpopulation
            (default 1.0).
        n_windows: Number of windows over which to complete the rotation for the affected
            subpopulation.
        seed: Random seed for selecting which samples to re-label within the affected subpopulation.
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X unchanged, y with concept drift applied to the
            selected subpopulation)
    """
    y_out = y.copy()
    if window_id >= start_window:
        rng = np.random.RandomState(seed if seed is not None else 0 + window_id)
        alpha = min(1.0, severity * (window_id - start_window) / n_windows)
        mask = X[condition_feature] > X[condition_feature].median()
        X_sub = X.loc[mask]
        fraction = alpha
        n_to_flip = int(fraction * len(X_sub))
        if n_to_flip > 0:
            score = (1 - alpha) * X_sub[original_feature] + alpha * X_sub[new_feature]
            new_labels = (score > score.median()).astype(int)
            sub_indices = X_sub.index.tolist()
            flip_indices = rng.choice(sub_indices, size=n_to_flip, replace=False)
            y_out.loc[flip_indices] = new_labels.loc[flip_indices]
    return X, y_out


def inject_mixed_drift(
    X: pd.DataFrame,
    y: pd.Series,
    window_id: int,
    covariate_feature: str,
    concept_original_feature: str,
    concept_new_feature: str,
    start_window: int,
    covariate_severity: float = 1.0,
    concept_severity: float = 1.0,
    n_windows: int = 20,
    seed: int | None = None,
    mode: str = "shift",
    concept_mode: str = "relabel",
    concept_condition_feature: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply both covariate and concept drift. Covariate drift is applied first, then concept
    drift is applied on top of the modified features.

    Args:
        X: Input DataFrame.
        y: Original labels (pd.Series).
        window_id: Current window index.
        covariate_feature: Column name to apply covariate drift to.
        concept_original_feature: Feature originally used for labeling (used in concept drift).
        concept_new_feature: Feature to rotate towards or use for re-labeling in concept drift.
        start_window: Window index to start applying drift.
        covariate_severity: Controls the magnitude of covariate drift (default 1.0).
        concept_severity: Controls the magnitude of concept drift (default 1.0
            for "relabel" and "rotation" modes).
        n_windows: Number of windows over which to increase the fraction of affected samples for
            concept drift.
        seed: Random seed for noise generation and sample selection.
        mode: Type of covariate drift to apply ("shift" or "noise").
        concept_mode: Type of concept drift to apply ("relabel", "rotation", or "conditional").
        concept_condition_feature: Required if concept_mode="conditional". Feature used to select
            the subpopulation affected by concept drift.
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X with covariate drift applied, y with
            concept drift applied)
    """
    X_out = inject_covariate_drift(X, window_id, covariate_feature,
                                   start_window, covariate_severity, seed, mode)
    if concept_mode == "relabel":
        _, y_out = inject_concept_drift(
            X_out, y, window_id, concept_original_feature,
            concept_new_feature, start_window, concept_severity, n_windows, seed,
        )
    elif concept_mode == "rotation":
        _, y_out = inject_concept_drift_rotation(
            X_out, y, window_id, concept_original_feature,
            concept_new_feature, start_window, concept_severity, n_windows, seed,
        )
    elif concept_mode == "conditional":
        if concept_condition_feature is None:
            raise ValueError(
                "concept_condition_feature is required when concept_mode='conditional'"
            )
        _, y_out = inject_concept_drift_conditional(
            X_out, y, window_id, concept_original_feature,
            concept_new_feature, concept_condition_feature, start_window,
            concept_severity, n_windows, seed,
        )
    else:
        raise ValueError(
            f"Unknown concept_mode: {concept_mode!r}. "
            "Use 'relabel', 'rotation', or 'conditional'."
        )
    return X_out, y_out
