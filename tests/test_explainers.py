"""Tests for SHAP, LIME, and Integrated Gradients explainers."""

import numpy as np
import pandas as pd
import torch
import pytest
from xgboost import XGBClassifier

from expl_drift.explanations.ig_explainer import explain_ig
from expl_drift.explanations.lime_explainer import explain_lime
from expl_drift.explanations.shap_explainer import explain_shap
from expl_drift_experiments.model.trainer import FeedforwardNN


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
    return X, y


@pytest.fixture
def xgb_model(synthetic_data):
    X, y = synthetic_data
    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42, eval_metric="logloss")
    model.fit(X, y)
    return model


@pytest.fixture
def nn_model(synthetic_data):
    X, y = synthetic_data
    torch.manual_seed(42)
    model = FeedforwardNN(input_dim=4, hidden_dims=(16, 8))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCELoss()
    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32)
    model.train()
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def test_shap_outputs_valid_shapes_and_finite_values(xgb_model, nn_model, synthetic_data):
    X, _ = synthetic_data

    attrs_xgb = explain_shap(xgb_model, X, X, model_type="xgboost")
    assert attrs_xgb.shape == (len(X), X.shape[1])
    assert np.all(np.isfinite(attrs_xgb))

    X_small = X.iloc[:10]
    attrs_nn = explain_shap(nn_model, X_small, X, model_type="nn")
    assert attrs_nn.shape == (len(X_small), X.shape[1])
    assert np.all(np.isfinite(attrs_nn))


def test_lime_outputs_shape_and_finite_values(xgb_model, synthetic_data):
    X, _ = synthetic_data
    attrs = explain_lime(
        xgb_model,
        X,
        X,
        feature_names=list(X.columns),
        model_type="xgboost",
        max_samples=20,
    )
    assert attrs.shape == (20, X.shape[1])
    assert np.all(np.isfinite(attrs))


def test_ig_outputs_shape_and_finite_values(nn_model, synthetic_data):
    X, _ = synthetic_data
    attrs = explain_ig(nn_model, X)
    assert attrs.shape == (len(X), X.shape[1])
    assert np.all(np.isfinite(attrs))


def test_ig_accepts_custom_baseline(nn_model, synthetic_data):
    X, _ = synthetic_data
    baseline = pd.DataFrame(np.ones_like(X.values), columns=X.columns)
    attrs = explain_ig(nn_model, X, baseline=baseline)
    assert attrs.shape == (len(X), X.shape[1])
