
"""
LiDAR Geometric Feature Extraction
"""

import argparse
import numpy as np
from pathlib import Path


def extract_features(points):
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must have shape (N, 3).")

    points = points[np.isfinite(points).all(axis=1)]

    if len(points) < 3:
        raise ValueError("At least 3 valid points are required.")

    centered = points - points.mean(axis=0)

    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues[::-1], 0)

    distances = np.linalg.norm(centered, axis=1)
    height = points[:, 2]

    features = {
        "point_count": len(points),

        "x_extent": np.ptp(points[:, 0]),
        "y_extent": np.ptp(points[:, 1]),
        "z_extent": np.ptp(points[:, 2]),

        "centroid_x": points[:, 0].mean(),
        "centroid_y": points[:, 1].mean(),
        "centroid_z": points[:, 2].mean(),

        "eigenvalue_1": eigenvalues[0],
        "eigenvalue_2": eigenvalues[1],
        "eigenvalue_3": eigenvalues[2],

        "radial_mean": distances.mean(),
        "radial_std": distances.std(),
        "radial_max": distances.max(),

        "height_mean": height.mean(),
        "height_std": height.std(),
        "height_min": height.min(),
        "height_max": height.max(),

        "height_q25": np.quantile(height, 0.25),
        "height_q50": np.quantile(height, 0.50),
        "height_q75": np.quantile(height, 0.75),
    }

    return features


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="geometric_features.csv"
    )

    args = parser.parse_args()

    points = np.load(args.input)
    features = extract_features(points)

    import pandas as pd

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([features]).to_csv(
        output, index=False
    )

    print("Feature extraction completed.")
    print(f"Features extracted: {len(features)}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
