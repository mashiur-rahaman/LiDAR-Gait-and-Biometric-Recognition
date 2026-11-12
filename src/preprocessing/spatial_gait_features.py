
"""
Spatial Gait Feature Extraction from LiDAR Point Clouds
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def extract_spatial_features(points):
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must have shape (N, 3).")

    points = points[np.isfinite(points).all(axis=1)]

    if len(points) < 3:
        raise ValueError("At least 3 valid points are required.")

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    covariance = np.cov(centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    eigenvectors = eigenvectors[:, order]

    total = eigenvalues.sum() + 1e-12

    features = {
        "point_count": len(points),

        "centroid_x": centroid[0],
        "centroid_y": centroid[1],
        "centroid_z": centroid[2],

        "extent_x": np.ptp(points[:, 0]),
        "extent_y": np.ptp(points[:, 1]),
        "extent_z": np.ptp(points[:, 2]),

        "eigenvalue_1": eigenvalues[0],
        "eigenvalue_2": eigenvalues[1],
        "eigenvalue_3": eigenvalues[2],

        "linearity": (
            eigenvalues[0] - eigenvalues[1]
        ) / (eigenvalues[0] + 1e-12),

        "planarity": (
            eigenvalues[1] - eigenvalues[2]
        ) / (eigenvalues[0] + 1e-12),

        "scattering": (
            eigenvalues[2]
        ) / (eigenvalues[0] + 1e-12),

        "anisotropy": (
            eigenvalues[0] - eigenvalues[2]
        ) / (eigenvalues[0] + 1e-12),

        "eigenvalue_entropy": -np.sum(
            (eigenvalues / total) *
            np.log(eigenvalues / total + 1e-12)
        ),

        "radial_distance_mean": np.linalg.norm(
            centered, axis=1
        ).mean(),

        "radial_distance_std": np.linalg.norm(
            centered, axis=1
        ).std(),

        "spatial_variance": total
    }

    for i in range(3):
        features[f"principal_axis_{i+1}_x"] = eigenvectors[0, i]
        features[f"principal_axis_{i+1}_y"] = eigenvectors[1, i]
        features[f"principal_axis_{i+1}_z"] = eigenvectors[2, i]

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Extract spatial features from LiDAR point clouds."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="spatial_features.csv"
    )

    args = parser.parse_args()

    points = np.load(args.input)

    features = extract_spatial_features(points)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([features]).to_csv(
        output,
        index=False
    )

    print("Spatial feature extraction completed.")
    print(f"Total features: {len(features)}")
    print(f"Output saved: {output}")


if __name__ == "__main__":
    main()
