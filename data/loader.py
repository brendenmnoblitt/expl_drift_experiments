"""Dataset loading and window partitioning helpers for experiments."""

import io
import logging

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

_ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]
_ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"


def _load_from_uci() -> tuple[pd.DataFrame, pd.Series]:
    """Fallback: download Adult dataset directly from UCI repository.
    
    This is a backup in case OpenML fetch fails. The UCI version has some missing values and may
    differ slightly from the OpenML version, but it allows experiments to proceed without
    interruption.
    
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    resp = requests.get(_ADULT_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(
        io.StringIO(resp.text),
        names=_ADULT_COLUMNS,
        sep=r",\s*",
        engine="python",
        na_values="?",
    )
    X = df.drop(columns=["income"]).copy()
    y = df["income"].copy()
    return X, y


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch UCI Adult Income dataset, clean it, and return (X, y).
    
    ~48K rows, 14 features. Binary target: income >50K or <=50K.
    Good drift features: age, education-num, hours-per-week (behavioral shifts).
    Reference: Kohavi (1996), data_id=1590 on OpenML.
    
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    try:
        data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        X = data.data.copy()
        y = data.target.copy()
    except Exception:
        X, y = _load_from_uci()

    # handle missing values: fill numeric with median, categorical with mode
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(exclude=["number"]).columns

    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    if len(cat_cols) > 0:
        X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

    # drop non-predictive and redundant features:
    # - fnlwgt: census sampling weight, not a real income predictor
    # - education: text version of education-num; after alphabetical label encoding
    #   its correlation with education-num is only ~0.36 (scrambles ordinal order).
    #   education-num is the correct ordinal representation.
    cols_to_drop = [c for c in ["fnlwgt", "education"] if c in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)

    # encode categorical features
    # NOTE: nominal categoricals (workclass, marital-status, etc.) are label-encoded
    # here for compatibility with tree models. One-hot encoding would improve SHAP
    # interpretability but is left as a future improvement.
    label_encoders: dict[str, LabelEncoder] = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    y = (y.astype(str).str.strip().str.rstrip(".")).map(
        lambda v: 1 if v == ">50K" else 0
    )
    y = y.astype(int)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y


def load_credit_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch UCI Default of Credit Card Clients dataset and return (X, y).

    ~30K rows, 23 features. Target: default payment next month (binary).
    Good drift features: AGE (demographic shift), LIMIT_BAL (credit limit distribution).
    Reference: Yeh & Lien (2009), data_id=42477 on OpenML.

    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    try:
        data = fetch_openml(data_id=42477, as_frame=True, parser="auto")
        X = data.data.copy()
        y = data.target.copy()
    except Exception:
        raise RuntimeError(
            "Could not fetch Default of Credit Card Clients (data_id=42477). "
            "Check network connectivity or OpenML availability."
        )

    # handle missing values
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(exclude=["number"]).columns
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    if len(cat_cols) > 0:
        X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

    # rncode any remaining categorical columns
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # rename generic x1-x23 columns to interpretable names (Yeh & Lien 2009)
    credit_col_map = {
        "x1": "LIMIT_BAL", "x2": "SEX", "x3": "EDUCATION", "x4": "MARRIAGE",
        "x5": "AGE",
        "x6": "PAY_0", "x7": "PAY_2", "x8": "PAY_3", "x9": "PAY_4",
        "x10": "PAY_5", "x11": "PAY_6",
        "x12": "BILL_AMT1", "x13": "BILL_AMT2", "x14": "BILL_AMT3",
        "x15": "BILL_AMT4", "x16": "BILL_AMT5", "x17": "BILL_AMT6",
        "x18": "PAY_AMT1", "x19": "PAY_AMT2", "x20": "PAY_AMT3",
        "x21": "PAY_AMT4", "x22": "PAY_AMT5", "x23": "PAY_AMT6",
    }
    X = X.rename(columns={k: v for k, v in credit_col_map.items() if k in X.columns})

    y = y.astype(str).str.strip().map(lambda v: 1 if v == "1" else 0).astype(int)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y


def load_electricity_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch the Electricity dataset and return (X, y) in chronological order.

    ~45,312 rows ordered by time. Binary target: UP/DOWN electricity price direction.
    Features: date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer.
    Well-studied drift benchmark (Harries 1999, Gama et al. 2004). OpenML data_id=151.
    
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y) with X in chronological order.
    """
    X: pd.DataFrame | None = None
    y: pd.Series | None = None
    for data_id in (151, 44120):
        try:
            data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
            X = data.data.copy()  # type: ignore[union-attr]
            y = data.target.copy()  # type: ignore[union-attr]
            break
        except Exception:
            if data_id == 44120:
                raise RuntimeError(
                    "Could not fetch Electricity dataset (tried data_id=151 and 44120). "
                    "Check network connectivity or OpenML availability."
                )
            continue
    assert X is not None and y is not None  # guaranteed by raise above

    # handle missing values
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(exclude=["number"]).columns
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    if len(cat_cols) > 0:
        X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

    # encode categorical features
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    y = y.astype(str).str.strip().map(lambda v: 1 if v.upper() == "UP" else 0).astype(int)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y


def partition_chronological(
    X: pd.DataFrame, y: pd.Series, n_windows: int = 20
) -> list[tuple[pd.DataFrame, pd.Series]]:
    """Split data into n_windows sequential batches preserving row order.

    Unlike partition_into_windows(), this does NOT shuffle the data.
    Designed for time-series datasets where chronological ordering matters.
    Window 0 is the earliest data (baseline/training window).
    
    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target labels.
        n_windows (int): Number of sequential windows to create.
    Returns:
        list[tuple[pd.DataFrame, pd.Series]]: List of (X_window, y_window) tuples.
    """
    window_size = len(X) // n_windows
    windows: list[tuple[pd.DataFrame, pd.Series]] = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else len(X)
        windows.append((X.iloc[start:end].copy(), y.iloc[start:end].copy()))
    return windows


def partition_into_windows(
    X: pd.DataFrame, y: pd.Series, n_windows: int = 20, seed: int | None = None
) -> list[tuple[pd.DataFrame, pd.Series]]:
    """Shuffle data and split into n_windows sequential batches.

    Window 0 is the baseline/training window.
    Returns list of (X_window, y_window) tuples.

    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target labels.
        n_windows (int): Number of sequential windows to create.
        seed (int | None): Random seed for shuffling.
    Returns:
        list[tuple[pd.DataFrame, pd.Series]]: List of (X_window, y_window) tuples.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    X_shuffled = X.iloc[indices].reset_index(drop=True)
    y_shuffled = y.iloc[indices].reset_index(drop=True)

    window_size = len(X_shuffled) // n_windows
    windows: list[tuple[pd.DataFrame, pd.Series]] = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else len(X_shuffled)
        windows.append((X_shuffled.iloc[start:end].copy(), y_shuffled.iloc[start:end].copy()))

    return windows


def get_baseline_window(
    windows: list[tuple[pd.DataFrame, pd.Series]],
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the baseline (training) window.
    
    Args:
        windows: List of (X_window, y_window) tuples.
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X_baseline, y_baseline)
    """
    return windows[0]


def get_window(
    windows: list[tuple[pd.DataFrame, pd.Series]], window_id: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a specific window by index.
    
    Args:
        windows: List of (X_window, y_window) tuples.
        window_id: Index of the window to return.
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X_window, y_window)
    """
    return windows[window_id]
