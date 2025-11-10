#!/usr/bin/env python3
"""
logistic_vs_xgb_demo.py

Train and compare a Logistic Regression model vs an XGBoost tree-based model
on the same synthetic classification dataset.

Features:
- Reuses the same synthetic dataset generation approach (make_classification).
- Splits data into train / val / test (stratified).
- Trains:
    - LogisticRegression (with StandardScaler in a pipeline)
    - XGBClassifier (scikit-learn wrapper) with optional early stopping
- Evaluates both models using: accuracy, precision, recall, f1, roc_auc
- Prints a side-by-side comparison and shows top XGBoost feature importances.

Dependencies:
- numpy, pandas, scikit-learn, xgboost

Example:
    python logistic_vs_xgb_demo.py --n-samples 2000 --n-features 12 --val-size 0.2

Author: GitHub Copilot-style assistant
"""

import argparse
import pprint
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier
import joblib

# ---------------------------
# Data generation (same style as previous XGBoost demo)
# ---------------------------
def generate_sample_classification(n_samples: int = 1000, n_features: int = 10, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(0, n_features // 10),
        n_clusters_per_class=1,
        flip_y=0.01,
        class_sep=1.0,
        random_state=rng
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_ser = pd.Series(y, name="target")
    return X_df, y_ser

def default_xgb_params() -> Dict:
    return {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }

# ---------------------------
# Training & evaluation helpers
# ---------------------------
def evaluate_classification(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            out["roc_auc"] = None
    else:
        out["roc_auc"] = None
    return out

def train_logistic(X_train, y_train, X_val=None, y_val=None, C=1.0, max_iter=1000):
    """
    Train logistic regression using a pipeline with standard scaling.
    If validation data provided, it is not used for early stopping (sklearn LogisticRegression
    does not have built-in early stopping for liblinear) but could be used for hyperparam tuning.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, solver="liblinear", max_iter=max_iter, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_xgb(X_train, y_train, X_val=None, y_val=None, params=None, early_stopping_rounds: int = 10):
    if params is None:
        params = default_xgb_params()
    model = XGBClassifier(**params)
    fit_kwargs = {}
    if X_val is not None:
        fit_kwargs["eval_set"] = [(X_train, y_train), (X_val, y_val)]
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    else:
        fit_kwargs["eval_set"] = [(X_train, y_train)]
    model.fit(X_train, y_train, **fit_kwargs)
    return model

# ---------------------------
# Main experiment orchestration
# ---------------------------
def run_experiment(n_samples: int = 1000,
                   n_features: int = 10,
                   test_size: float = 0.2,
                   val_size: float = 0.2,
                   xgb_early_stopping: int = 10,
                   save_dir: str = "models"):
    # 1) Generate data
    X, y = generate_sample_classification(n_samples=n_samples, n_features=n_features)
    print(f"Generated dataset with {n_samples} samples and {n_features} features.")
    print("Target distribution:")
    print(y.value_counts().to_string())

    # 2) Split into train+val / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # carve out validation from trainval
    rel_val = val_size / (1 - test_size) if val_size and val_size > 0 else 0.0
    if rel_val > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=rel_val, random_state=42, stratify=y_trainval
        )
        print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    else:
        X_train, y_train = X_trainval, y_trainval
        X_val, y_val = None, None
        print(f"Split: train+val={len(X_train)}, test={len(X_test)} (no validation set)")

    # 3) Train Logistic Regression
    print("\nTraining Logistic Regression...")
    log_pipe = train_logistic(X_train, y_train, X_val, y_val)
    y_pred_log = log_pipe.predict(X_test)
    y_proba_log = log_pipe.predict_proba(X_test)[:, 1]
    metrics_log = evaluate_classification(y_test, y_pred_log, y_proba_log)
    print("Logistic Regression metrics:")
    pprint.pprint(metrics_log)

    # 4) Train XGBoost
    print("\nTraining XGBoost classifier...")
    xgb_params = default_xgb_params()
    xgb_model = train_xgb(X_train, y_train, X_val, y_val, params=xgb_params, early_stopping_rounds=xgb_early_stopping)
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1] if hasattr(xgb_model, "predict_proba") else None
    metrics_xgb = evaluate_classification(y_test, y_pred_xgb, y_proba_xgb)
    print("XGBoost metrics:")
    pprint.pprint(metrics_xgb)

    # 5) Feature importances (XGBoost) and coefficient magnitudes (Logistic)
    fi_xgb = None
    try:
        fi_xgb = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    except Exception:
        fi_xgb = pd.Series(dtype=float)

    coef_log = None
    try:
        # extract coefficients from pipeline
        coef = log_pipe.named_steps["clf"].coef_.ravel()
        coef_log = pd.Series(np.abs(coef), index=X.columns).sort_values(ascending=False)
    except Exception:
        coef_log = pd.Series(dtype=float)

    # 6) Save models
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "logistic_model.joblib")
    xgb_path = os.path.join(save_dir, "xgb_model.joblib")
    joblib.dump(log_pipe, log_path)
    joblib.dump(xgb_model, xgb_path)

    # 7) Summary comparison
    print("\n--- Summary Comparison (higher is better unless stated) ---")
    headers = ["metric", "logistic", "xgboost"]
    metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comp_rows = []
    for m in metrics_list:
        comp_rows.append((m, metrics_log.get(m), metrics_xgb.get(m)))
    # Print compact table-like output
    print(f"{headers[0]:<12} {headers[1]:<12} {headers[2]:<12}")
    for r in comp_rows:
        print(f"{r[0]:<12} {str(round(r[1],4)) if r[1] is not None else 'None':<12} {str(round(r[2],4)) if r[2] is not None else 'None':<12}")

    print("\nTop XGBoost feature importances (top 10):")
    if not fi_xgb.empty:
        print(fi_xgb.head(10).to_string())
    else:
        print("No XGBoost feature importances available.")

    print("\nTop Logistic coefficient magnitudes (top 10):")
    if not coef_log.empty:
        print(coef_log.head(10).to_string())
    else:
        print("No logistic coefficients available.")

    print(f"\nSaved logistic model to: {log_path}")
    print(f"Saved xgboost model to:   {xgb_path}")

    # Return results for programmatic consumption if imported
    return {
        "logistic": {"model": log_pipe, "metrics": metrics_log, "coef_magnitudes": coef_log},
        "xgboost": {"model": xgb_model, "metrics": metrics_xgb, "feature_importances": fi_xgb},
        "X_test": X_test, "y_test": y_test
    }

def parse_args():
    p = argparse.ArgumentParser(description="Compare Logistic Regression vs XGBoost on synthetic data")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-features", type=int, default=10)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--xgb-early-stopping", type=int, default=10)
    p.add_argument("--save-dir", type=str, default="models")
    return p.parse_args()

def main():
    args = parse_args()
    run_experiment(
        n_samples=args.n_samples,
        n_features=args.n_features,
        test_size=args.test_size,
        val_size=args.val_size,
        xgb_early_stopping=args.xgb_early_stopping,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()