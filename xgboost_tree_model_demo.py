#!/usr/bin/env python3
"""
xgboost_tree_model_demo.py

A self-contained demo to build a tree-based model with the XGBoost algorithm
using a synthetic sample dataset. Supports both classification and regression.

Features:
- Generates a synthetic dataset (make_classification / make_regression)
- Train / validation split and optional early stopping
- Uses XGBoost's sklearn API (XGBClassifier / XGBRegressor)
- Basic hyperparameters example and feature importance
- Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC for classification;
  RMSE, MAE, R2 for regression)
- Save / load model example

Dependencies:
- numpy
- pandas
- scikit-learn
- xgboost
- joblib (for model saving) or xgboost native save_model

Install dependencies (example):
    pip install numpy pandas scikit-learn xgboost joblib

Usage examples:
    python xgboost_tree_model_demo.py --task classification
    python xgboost_tree_model_demo.py --task regression --n-samples 5000 --n-features 20

Author: Copilot-style assistant
"""

import argparse
import pprint
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
import joblib


def generate_sample_data(task: str, n_samples: int = 1000, n_features: int = 10, random_state: int = 42):
    """
    Generate a synthetic dataset for classification or regression.
    Returns (X: pd.DataFrame, y: pd.Series)
    """
    rng = np.random.RandomState(random_state)
    if task == "classification":
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
    elif task == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            noise=0.1,
            random_state=rng
        )
    else:
        raise ValueError("Unsupported task. Choose 'classification' or 'regression'.")

    # Put into DataFrame for nicer feature names
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_ser = pd.Series(y, name="target")
    return X_df, y_ser


def default_xgb_params(task: str) -> Dict:
    """
    Return a small set of sensible default parameters for XGBoost.
    You can expand or tune these for better performance.
    """
    base = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
    }
    if task == "classification":
        base.update({
            "use_label_encoder": False,
            "eval_metric": "logloss",
        })
    else:
        base.update({
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        })
    return base


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    early_stopping_rounds: int = 10,
    params: Dict = None
):
    """
    Train an XGBoost model (classifier or regressor) and evaluate on a holdout test set.
    Returns: dict with model, metrics, feature_importances_
    """
    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=(y if task == "classification" else None)
    )

    # From trainval, carve out a validation set for early stopping
    if val_size and val_size > 0:
        # relative to original dataset: val_size is fraction of entire data; compute fraction relative to trainval
        rel_val = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=rel_val, random_state=42, stratify=(y_trainval if task == "classification" else None)
        )
    else:
        X_train, y_train = X_trainval, y_trainval
        X_val, y_val = None, None

    if params is None:
        params = default_xgb_params(task)

    # Choose estimator
    if task == "classification":
        model = XGBClassifier(**params)
    else:
        model = XGBRegressor(**params)

    fit_kwargs = {}
    eval_set = []
    if X_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        fit_kwargs["eval_set"] = eval_set
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    else:
        eval_set = [(X_train, y_train)]
        fit_kwargs["eval_set"] = eval_set

    # Fit
    model.fit(X_train, y_train, **fit_kwargs)

    # Predict and evaluate
    results = {}
    if task == "classification":
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        results["accuracy"] = accuracy_score(y_test, y_pred)
        results["precision"] = precision_score(y_test, y_pred, zero_division=0)
        results["recall"] = recall_score(y_test, y_pred, zero_division=0)
        results["f1"] = f1_score(y_test, y_pred, zero_division=0)
        if y_proba is not None:
            try:
                results["roc_auc"] = roc_auc_score(y_test, y_proba)
            except Exception:
                results["roc_auc"] = None
        else:
            results["roc_auc"] = None

    else:  # regression
        y_pred = model.predict(X_test)
        results["rmse"] = mean_squared_error(y_test, y_pred, squared=False)
        results["mae"] = mean_absolute_error(y_test, y_pred)
        results["r2"] = r2_score(y_test, y_pred)

    # Feature importance (gain-based if available)
    try:
        fmap = X.columns.tolist()
        # model.feature_importances_ is available for sklearn wrapper; it's based on weight by default
        fi = model.feature_importances_
        fi_series = pd.Series(fi, index=fmap).sort_values(ascending=False)
    except Exception:
        fi_series = pd.Series(dtype=float)

    # Collect model info
    out = {
        "model": model,
        "metrics": results,
        "feature_importances": fi_series,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }
    if task == "classification":
        out["y_proba"] = y_proba

    return out


def save_model(model, path: str):
    """
    Save model using joblib for sklearn wrapper objects.
    """
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    joblib.dump(model, path)
    return path


def load_model(path: str):
    """
    Load model saved by save_model.
    """
    return joblib.load(path)


def parse_args():
    p = argparse.ArgumentParser(description="XGBoost tree-based demo for classification/regression")
    p.add_argument("--task", choices=["classification", "regression"], default="classification",
                   help="Task type: classification (default) or regression")
    p.add_argument("--n-samples", type=int, default=1000, help="Number of synthetic samples to generate")
    p.add_argument("--n-features", type=int, default=10, help="Number of features")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    p.add_argument("--val-size", type=float, default=0.2, help="Validation split fraction (of full data). Set 0 to disable")
    p.add_argument("--early-stopping-rounds", type=int, default=10, help="Early stopping rounds; 0 to disable")
    p.add_argument("--save-model", type=str, default="xgb_model.joblib", help="Path to save trained model")
    return p.parse_args()


def main():
    args = parse_args()

    # Step 1: generate data
    X, y = generate_sample_data(args.task, n_samples=args.n_samples, n_features=args.n_features)
    print("Generated dataset:")
    print(f" - task: {args.task}")
    print(f" - samples: {args.n_samples}, features: {args.n_features}")
    print(" - head of X:")
    print(X.head().to_string(index=False))
    print(" - target distribution (classification) / sample stats (regression):")
    if args.task == "classification":
        print(y.value_counts().to_string())
    else:
        print(y.describe().to_string())

    # Step 2: train & evaluate
    early_stopping = args.early_stopping_rounds if args.early_stopping_rounds > 0 else None
    result = train_and_evaluate(
        X, y, task=args.task,
        test_size=args.test_size,
        val_size=args.val_size,
        early_stopping_rounds=early_stopping
    )

    # Step 3: report metrics and feature importances
    print("\nTraining completed. Evaluation metrics:")
    pprint.pprint(result["metrics"])

    print("\nTop feature importances:")
    fi = result["feature_importances"]
    if not fi.empty:
        print(fi.head(20).to_string())
    else:
        print("No feature importances available.")

    # Step 4: save model
    save_path = args.save_model
    save_model(result["model"], save_path)
    print(f"\nSaved trained model to: {save_path}")

    # Step 5: demo predict on a few test rows
    print("\nPredictions on first 5 test rows:")
    X_test = result["X_test"].iloc[:5]
    preds = result["model"].predict(X_test)
    out_df = X_test.copy()
    out_df["pred"] = preds
    if args.task == "classification" and "y_proba" in result:
        proba = result["model"].predict_proba(X_test)[:, 1]
        out_df["proba_pos"] = proba
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()