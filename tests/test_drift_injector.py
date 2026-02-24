"""Tests for synthetic drift injection operators."""

import numpy as np
import pandas as pd
import pytest

from expl_drift_experiments.data.drift_injector import (
    inject_concept_drift,
    inject_concept_drift_conditional,
    inject_concept_drift_perturbation,
    inject_concept_drift_rotation,
    inject_covariate_drift,
    inject_mixed_drift,
)


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({
        "feat_a": rng.randn(100),
        "feat_b": rng.randn(100),
        "feat_c": rng.randn(100),
    })
    y = pd.Series(rng.randint(0, 2, 100))
    return X, y


def test_covariate_shift_is_deterministic_and_non_mutating(sample_data):
    X, _ = sample_data
    X_copy = X.copy()
    X_out = inject_covariate_drift(
        X, window_id=8, feature="feat_a", start_window=5, severity=1.0, mode="shift"
    )
    expected_shift = 3.0
    np.testing.assert_allclose(X_out["feat_a"].values, X["feat_a"].values + expected_shift)
    pd.testing.assert_frame_equal(X, X_copy)


def test_covariate_noise_mode_deterministic_with_seed(sample_data):
    X, _ = sample_data
    X_out1 = inject_covariate_drift(
        X, window_id=8, feature="feat_a", start_window=5, seed=42, mode="noise"
    )
    X_out2 = inject_covariate_drift(
        X, window_id=8, feature="feat_a", start_window=5, seed=42, mode="noise"
    )
    pd.testing.assert_frame_equal(X_out1, X_out2)


def test_concept_drift_noop_before_start_and_deterministic_after_start(sample_data):
    X, y = sample_data
    _, y_pre = inject_concept_drift(
        X, y, window_id=2, original_feature="feat_a", new_feature="feat_b", start_window=5
    )
    pd.testing.assert_series_equal(y_pre, y)

    _, y1 = inject_concept_drift(
        X, y, window_id=10, original_feature="feat_a", new_feature="feat_b", start_window=5, seed=42
    )
    _, y2 = inject_concept_drift(
        X, y, window_id=10, original_feature="feat_a", new_feature="feat_b", start_window=5, seed=42
    )
    pd.testing.assert_series_equal(y1, y2)


def test_rotation_full_alpha_uses_new_feature_rule(sample_data):
    X, y = sample_data
    _, y_out = inject_concept_drift_rotation(
        X,
        y,
        window_id=25,
        original_feature="feat_a",
        new_feature="feat_b",
        start_window=5,
        severity=1.0,
        n_windows=20,
    )
    expected = (X["feat_b"] > X["feat_b"].median()).astype(int)
    pd.testing.assert_series_equal(y_out, expected, check_names=False)


def test_perturbation_is_deterministic_for_fixed_seed(sample_data):
    X, y = sample_data
    _, y1 = inject_concept_drift_perturbation(
        X, y, window_id=10, features=["feat_a", "feat_b"], start_window=5, seed=42
    )
    _, y2 = inject_concept_drift_perturbation(
        X, y, window_id=10, features=["feat_a", "feat_b"], start_window=5, seed=42
    )
    pd.testing.assert_series_equal(y1, y2)


def test_conditional_drift_only_changes_conditioned_subpopulation(sample_data):
    X, y = sample_data
    _, y_out = inject_concept_drift_conditional(
        X,
        y,
        window_id=15,
        original_feature="feat_a",
        new_feature="feat_b",
        condition_feature="feat_c",
        start_window=5,
        severity=2.0,
    )
    below_median = X["feat_c"] <= X["feat_c"].median()
    pd.testing.assert_series_equal(y_out[below_median], y[below_median])


def test_mixed_drift_applies_both_covariate_and_concept(sample_data):
    X, y = sample_data
    X_out, y_out = inject_mixed_drift(
        X,
        y,
        window_id=10,
        covariate_feature="feat_a",
        concept_original_feature="feat_a",
        concept_new_feature="feat_b",
        start_window=5,
        concept_mode="rotation",
    )
    assert not np.allclose(X_out["feat_a"].values, X["feat_a"].values)
    assert not y_out.equals(y)


def test_mixed_conditional_requires_condition_feature(sample_data):
    X, y = sample_data
    with pytest.raises(ValueError, match="concept_condition_feature is required"):
        inject_mixed_drift(
            X,
            y,
            window_id=10,
            covariate_feature="feat_a",
            concept_original_feature="feat_a",
            concept_new_feature="feat_b",
            start_window=5,
            concept_mode="conditional",
        )


def test_mixed_default_mode_matches_explicit_relabel(sample_data):
    X, y = sample_data
    X_out1, y_out1 = inject_mixed_drift(
        X,
        y,
        window_id=10,
        covariate_feature="feat_a",
        concept_original_feature="feat_a",
        concept_new_feature="feat_b",
        start_window=5,
        seed=42,
    )
    X_out2, y_out2 = inject_mixed_drift(
        X,
        y,
        window_id=10,
        covariate_feature="feat_a",
        concept_original_feature="feat_a",
        concept_new_feature="feat_b",
        start_window=5,
        seed=42,
        concept_mode="relabel",
    )
    pd.testing.assert_frame_equal(X_out1, X_out2)
    pd.testing.assert_series_equal(y_out1, y_out2)
