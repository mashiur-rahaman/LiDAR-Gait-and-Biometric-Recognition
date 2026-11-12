
"""
LiDAR Point Cloud Preprocessing
LiDAR-Gait-and-Biometric-Recognition

Supports PLY, PCD, XYZ, TXT, CSV, NPY and NPZ.
"""

from pathlib import Path
import argparse
import numpy as np
import open3d as o3d


def load_point_cloud(file_path):
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in [".ply", ".pcd", ".xyz"]:
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)

    elif suffix == ".npy":
        points = np.load(path)

    elif suffix == ".npz":
        data = np.load(path)
        points = data["points"]

    elif suffix in [".txt", ".csv"]:
        points = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)

    else:
        raise ValueError(f"Unsupported format: {suffix}")

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Point cloud must have at least 3 columns.")

    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]

    if len(points) == 0:
        raise ValueError("No valid points found.")

    return points


def remove_outliers(points, nb_neighbors=20, std_ratio=2.0):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    filtered, indices = cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    return np.asarray(filtered.points)


def normalize_point_cloud(points, normalize_scale=False):
    center = np.mean(points, axis=0)
    points = points - center

    if normalize_scale:
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale

    return points


def preprocess(
    input_path,
    output_path,
    voxel_size=0.02,
    normalize_scale=False
):
    points = load_point_cloud(input_path)
    original_count = len(points)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)

    points = np.asarray(cloud.points)

    if len(points) > 20:
        points = remove_outliers(points)

    points = normalize_point_cloud(
        points,
        normalize_scale=normalize_scale
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, points)

    print("Preprocessing completed.")
    print(f"Original points: {original_count}")
    print(f"Processed points: {len(points)}")
    print(f"Output: {output_path}")

    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LiDAR point cloud preprocessing"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="processed_cloud.npy")
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--normalize_scale", action="store_true")

    args = parser.parse_args()

    preprocess(
        args.input,
        args.output,
        args.voxel_size,
        args.normalize_scale
    )
