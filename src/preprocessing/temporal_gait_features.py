
"""
Temporal Gait Feature Extraction from LiDAR Sequences
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def extract_temporal_features(sequence, fps=30):
    sequence = np.asarray(sequence, dtype=np.float64)

    if sequence.ndim != 3 or sequence.shape[2] != 3:
        raise ValueError("Expected input shape (T, N, 3).")

    if fps <= 0:
        raise ValueError("FPS must be positive.")

    frame_features = []

    for frame in sequence:
        points = frame[np.isfinite(frame).all(axis=1)]

        if len(points) < 3:
            raise ValueError("Each frame needs at least 3 valid points.")

        centroid = points.mean(axis=0)
        centered = points - centroid

        eigenvalues = np.linalg.eigvalsh(
            np.cov(centered, rowvar=False)
        )[::-1]

        frame_features.append([
            *centroid,
            *np.ptp(points, axis=0),
            *eigenvalues,
            np.linalg.norm(centered, axis=1).mean()
        ])

    features = np.asarray(frame_features)

    velocity = np.diff(features, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps

    result = {}

    names = [
        "centroid_x", "centroid_y", "centroid_z",
        "extent_x", "extent_y", "extent_z",
        "eigenvalue_1", "eigenvalue_2",
        "eigenvalue_3", "radial_mean"
    ]

    for i, name in enumerate(names):
        result[f"{name}_mean"] = features[:, i].mean()
        result[f"{name}_std"] = features[:, i].std()

    for i, name in enumerate(names):
        result[f"{name}_velocity_mean"] = (
            np.abs(velocity[:, i]).mean()
            if len(velocity) else 0.0
        )

        result[f"{name}_acceleration_mean"] = (
            np.abs(acceleration[:, i]).mean()
            if len(acceleration) else 0.0
        )

    result["sequence_length"] = len(sequence)
    result["duration_seconds"] = len(sequence) / fps

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="temporal_features.csv")
    parser.add_argument("--fps", type=float, default=30)

    args = parser.parse_args()

    sequence = np.load(args.input)
    result = extract_temporal_features(sequence, args.fps)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([result]).to_csv(output, index=False)

    print("Temporal gait feature extraction completed.")
    print(f"Features: {len(result)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
