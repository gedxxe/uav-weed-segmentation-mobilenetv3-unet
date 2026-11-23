"""Synthetic CV tests. These tests do not launch GPU training."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.research_train import normalize_stats, split_grouped


class GroupedSplitTests(unittest.TestCase):
    def test_groups_never_overlap_and_validation_is_unfiltered(self):
        rows = []
        for source in ("A", "B", "C", "D"):
            for patch in range(2):
                rows.append({
                    "source_image": source,
                    "patch_image": f"{source}_{patch}.png",
                    "patch_mask": f"{source}_{patch}_msk.png",
                    "train_eligible": "1" if patch == 0 else "0",
                })
        folds = list(split_grouped(rows, 2))
        self.assertEqual(len(folds), 2)
        for fold, training, validation, train_sources, val_sources in folds:
            self.assertFalse(train_sources & val_sources)
            self.assertTrue(all(int(r["train_eligible"]) == 1 for r in training))
            self.assertEqual(len(validation), len(val_sources) * 2)

    def test_normalization_uses_only_valid_training_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pixels = np.zeros((4, 4, 3), dtype=np.uint8)
            pixels[:2, :2, 2] = 255  # OpenCV BGR: red
            path = root / "tile.png"
            cv2.imwrite(str(path), pixels)
            mean, std = normalize_stats(root, [{
                "patch_image": "tile.png", "valid_height": "2", "valid_width": "2",
            }])
            self.assertTrue(np.allclose(mean, [1.0, 0.0, 0.0]))
            self.assertTrue(all(s > 0 for s in std))


if __name__ == "__main__":
    unittest.main()
