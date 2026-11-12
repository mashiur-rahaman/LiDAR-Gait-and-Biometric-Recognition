
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "feature_extraction"))

from spatial_gait_features import extract_spatial_features
from temporal_gait_features import extract_temporal_features
from local_geometric_features import extract_local_features


class TestGaitFeatureExtraction(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        self.points = rng.normal(size=(200, 3))

    def test_spatial_features(self):
        features = extract_spatial_features(self.points)

        self.assertIn("centroid_x", features)
        self.assertIn("eigenvalue_1", features)
        self.assertTrue(
            np.isfinite(list(features.values())).all()
        )

    def test_temporal_features(self):
        sequence = np.stack([
            self.points + i * 0.01
            for i in range(5)
        ])

        features = extract_temporal_features(
            sequence, fps=30
        )

        self.assertIn("duration_seconds", features)
        self.assertTrue(
            np.isfinite(list(features.values())).all()
        )

    def test_local_geometric_features(self):
        features = extract_local_features(
            self.points,
            radius=3.0,
            min_neighbors=5
        )

        self.assertEqual(features.shape, (200, 7))
        self.assertTrue(np.isfinite(features).all())

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            extract_spatial_features(
                np.array([[1, 2]])
            )


if __name__ == "__main__":
    unittest.main()
