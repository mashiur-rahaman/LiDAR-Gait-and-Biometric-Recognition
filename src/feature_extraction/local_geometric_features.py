
"""
Local Geometric Feature Extraction for LiDAR Point Clouds
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree


def extract_local_features(points, radius=0.3, min_neighbors=5):

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must have shape (N, 3).")

    points = points[np.isfinite(points).all(axis=1)]

    if len(points) < min_neighbors:
        raise ValueError("Insufficient valid points.")

    if radius <= 0:
        raise ValueError("Radius must be positive.")

    tree = cKDTree(points)
    neighborhoods = tree.query_ball_point(points, radius)

    features = []

    for i, indices in enumerate(neighborhoods):

        if len(indices) < min_neighbors:
            features.append([0.0] * 7)
            continue

        local_points = points[indices]
        centered = local_points - local_points.mean(axis=0)

        covariance = np.cov(centered, rowvar=False)

        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.maximum(eigenvalues[::-1], 0)

        l1, l2, l3 = eigenvalues
        eps = 1e-12

        linearity = (l1 - l2) / (l1 + eps)
        planarity = (l2 - l3) / (l1 + eps)
        scattering = l3 / (l1 + eps)

        curvature = l3 / (l1 + l2 + l3 + eps)

        density = len(indices) / (
            (4.0 / 3.0) * np.pi * radius ** 3
        )

        features.append([
            l1, l2, l3,
            linearity,
            planarity,
            scattering,
            curvature
        ])

    return np.asarray(features, dtype=np.float32)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="local_geometric_features.npy"
    )
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--min-neighbors", type=int, default=5)

    args = parser.parse_args()

    points = np.load(args.input)

    features = extract_local_features(
        points,
        radius=args.radius,
        min_neighbors=args.min_neighbors
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.save(output, features)

    print("Local geometric feature extraction completed.")
    print("Input points:", len(points))
    print("Feature shape:", features.shape)
    print("Saved:", output)


if __name__ == "__main__":
    main()
