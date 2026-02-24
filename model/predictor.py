"""Prediction and window-level evaluation helpers."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def predict_batch(
    model: object, X: np.ndarray, model_type: str = "xgboost"
) -> tuple[np.ndarray, np.ndarray]:
    """Make predictions and compute probabilities for a batch of data using the provided model.
    
    Args:
        model: Trained model.
        X: Input data.
        model_type: Type of model (e.g. "xgboost", "nn").
    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (predictions, probabilities).
    """
    if model_type == "xgboost":
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
    elif model_type == "nn":
        model.eval()
        X_tensor = torch.tensor(
            X.values if hasattr(X, "values") else X, dtype=torch.float32
        )
        with torch.no_grad():
            probabilities = model(X_tensor).numpy()
        predictions = (probabilities >= 0.5).astype(int)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return predictions, probabilities


def evaluate_window(
    model: object, X: np.ndarray, y: np.ndarray, model_type: str = "xgboost"
) -> dict[str, float]:
    """Evaluate model performance on a single window of data, returning accuracy,
    precision, recall, and F1 score.
    
    Args:
        model: Trained model.
        X: Input data.
        y: True labels.
        model_type: Type of model (e.g. "xgboost", "nn").
    Returns:
        dict[str, float]: Dictionary with keys "accuracy", "precision", "recall", "f1".
    """
    predictions, _ = predict_batch(model, X, model_type)
    y_np = y.values if hasattr(y, "values") else np.asarray(y)
    return {
        "accuracy": accuracy_score(y_np, predictions),
        "precision": precision_score(y_np, predictions, zero_division=0),
        "recall": recall_score(y_np, predictions, zero_division=0),
        "f1": f1_score(y_np, predictions, zero_division=0),
    }
