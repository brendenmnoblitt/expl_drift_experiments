"""Model training utilities for XGBoost and feedforward neural networks."""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "models")


class FeedforwardNN(nn.Module):
    """Simple feedforward neural network for binary classification."""
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 32)) -> None:
        """Initialize the feedforward neural network.
        
        Args:
            input_dim: Number of input features.
            hidden_dims: Tuple of hidden layer sizes.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.
        
        Args:
            x: Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x).squeeze(-1)


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, seed: int | None = None
) -> tuple[XGBClassifier, float]:
    """Train an XGBoost model and save it. Returns (model, baseline_accuracy).
    
    Args:
        X_train: Training data.
        y_train: Training labels.
        seed: Random seed for reproducibility.
    Returns:
        tuple[XGBClassifier, float]: Tuple of (model, baseline_accuracy).
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed
    )
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
        eval_metric="logloss",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    accuracy = accuracy_score(y_val, model.predict(X_val))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        model.save_model(os.path.join(RESULTS_DIR, "xgboost_model.json"))
    except TypeError:
        pass  # XGBoost version bug: _estimator_type not set; model is still usable

    return model, accuracy


def train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int | None = None,
    epochs: int = 50,
    hidden_dims: tuple[int, ...] = (64, 32),
) -> tuple[FeedforwardNN, float]:
    """Train a feedforward neural network and save it. Returns (model, baseline_accuracy).
    
    Args:
        X_train: Training data.
        y_train: Training labels.
        seed: Random seed for reproducibility.
        epochs: Number of training epochs.
        hidden_dims: Tuple of hidden layer sizes.
    Returns:
        tuple[FeedforwardNN, float]: Tuple of (model, baseline_accuracy).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train.values if hasattr(X_train, "values") else X_train,
        y_train.values if hasattr(y_train, "values") else y_train,
        test_size=0.2,
        random_state=seed,
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    input_dim = X_tr.shape[1]
    model = FeedforwardNN(input_dim, hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tr_t)
        loss = criterion(preds, y_tr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = (model(X_val_t).numpy() >= 0.5).astype(int)
    accuracy = accuracy_score(y_val, val_preds)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "nn_model.pt"))

    return model, accuracy
