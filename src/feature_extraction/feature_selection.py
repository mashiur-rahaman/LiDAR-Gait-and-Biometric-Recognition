
"""
Feature Selection for LiDAR-Based Gait Recognition
"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer


def select_features(X, y, k=20):

    if X.ndim != 2 or len(X) == 0:
        raise ValueError("Invalid feature matrix.")

    if len(np.unique(y)) < 2:
        raise ValueError("At least two classes are required.")

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinite values.")

    k = min(k, X.shape[1])

    if k < 1:
        raise ValueError("k must be positive.")

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    X_selected = selector.fit_transform(X, y)

    return X_selected, selector


def main():

    parser = argparse.ArgumentParser(
        description="Select discriminative gait features."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output", default="selected_features.csv")
    parser.add_argument("--model", default="feature_selector.joblib")
    parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.label_column not in df.columns:
        raise ValueError(
            f"Label column '{args.label_column}' not found."
        )

    y = df[args.label_column].to_numpy()

    X_df = df.drop(columns=[args.label_column])

    if X_df.empty:
        raise ValueError("No feature columns found.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in X_df.dtypes
    ):
        raise ValueError("All feature columns must be numeric.")

    if pd.isna(y).any():
        raise ValueError("Labels contain missing values.")

    X = X_df.to_numpy(dtype=float)

    if np.isinf(X).any():
        raise ValueError("Features contain infinite values.")

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    selected, selector = select_features(
        X, y, args.top_k
    )

    selected_names = X_df.columns[
        selector.get_support()
    ].tolist()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame(
        selected,
        columns=selected_names
    )

    result[args.label_column] = y
    result.to_csv(output, index=False)

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "imputer": imputer,
            "selector": selector,
            "selected_features": selected_names
        },
        model_path
    )

    print("Feature selection completed.")
    print("Original features:", X_df.shape[1])
    print("Selected features:", len(selected_names))
    print("Selected names:", selected_names)
    print("Output:", output)
    print("Saved model:", model_path)


if __name__ == "__main__":
    main()
