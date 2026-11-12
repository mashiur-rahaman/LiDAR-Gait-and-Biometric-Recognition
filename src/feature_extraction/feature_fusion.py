
"""
Feature Fusion Module for LiDAR-Based Gait Recognition
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def load_single_row_csv(path, prefix):
    df = pd.read_csv(path)

    if len(df) != 1:
        raise ValueError(
            f"{path} must contain exactly one feature row."
        )

    if df.empty or not len(df.columns):
        raise ValueError(f"Empty feature file: {path}")

    df.columns = [f"{prefix}_{c}" for c in df.columns]

    return df.reset_index(drop=True)


def summarize_local_features(features):
    features = np.asarray(features, dtype=np.float64)

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("Local features must be a non-empty 2D array.")

    if not np.isfinite(features).all():
        raise ValueError("Local features contain invalid values.")

    result = {}

    for i in range(features.shape[1]):
        column = features[:, i]

        result[f"local_{i}_mean"] = np.mean(column)
        result[f"local_{i}_std"] = np.std(column)
        result[f"local_{i}_median"] = np.median(column)
        result[f"local_{i}_min"] = np.min(column)
        result[f"local_{i}_max"] = np.max(column)

    return pd.DataFrame([result])


def fuse_features(spatial_path, temporal_path, local_path):

    spatial = load_single_row_csv(
        spatial_path, "spatial"
    )

    temporal = load_single_row_csv(
        temporal_path, "temporal"
    )

    local_array = np.load(local_path)

    local = summarize_local_features(local_array)

    fused = pd.concat(
        [spatial, temporal, local],
        axis=1
    )

    if not fused.columns.is_unique:
        raise ValueError("Duplicate feature names detected.")

    if not np.isfinite(
        fused.select_dtypes(include=[np.number]).to_numpy()
    ).all():
        raise ValueError("Non-finite values detected.")

    return fused


def main():

    parser = argparse.ArgumentParser(
        description="Fuse LiDAR gait recognition features."
    )

    parser.add_argument("--spatial", required=True)
    parser.add_argument("--temporal", required=True)
    parser.add_argument("--local", required=True)

    parser.add_argument(
        "--output",
        default="fused_gait_features.csv"
    )

    args = parser.parse_args()

    fused = fuse_features(
        args.spatial,
        args.temporal,
        args.local
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fused.to_csv(output, index=False)

    print("Feature fusion completed successfully.")
    print("Total fused features:", fused.shape[1])
    print("Output:", output)


if __name__ == "__main__":
    main()
