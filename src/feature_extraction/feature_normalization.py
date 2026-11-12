
"""
Feature Normalization for LiDAR-Based Gait Recognition
"""

import argparse
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def normalize_features(data, method="standard"):

    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("Input must be a non-empty 2D array.")

    if not np.isfinite(data).all():
        raise ValueError("Input contains NaN or infinite values.")

    if method == "standard":
        scaler = StandardScaler()

    elif method == "minmax":
        scaler = MinMaxScaler()

    else:
        raise ValueError("Method must be standard or minmax.")

    normalized = scaler.fit_transform(data)

    return normalized, scaler


def main():

    parser = argparse.ArgumentParser(
        description="Normalize fused gait features."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="normalized_features.csv")
    parser.add_argument("--scaler", default="feature_scaler.joblib")
    parser.add_argument(
        "--method",
        choices=["standard", "minmax"],
        default="standard"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if df.empty:
        raise ValueError("Input feature file is empty.")

    if not df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]:
        raise ValueError("All input columns must be numeric.")

    normalized, scaler = normalize_features(
        df.values,
        args.method
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        normalized,
        columns=df.columns
    ).to_csv(output, index=False)

    scaler_path = Path(args.scaler)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, scaler_path)

    print("Feature normalization completed.")
    print("Method:", args.method)
    print("Feature shape:", normalized.shape)
    print("Saved:", output)
    print("Scaler:", scaler_path)


if __name__ == "__main__":
    main()
