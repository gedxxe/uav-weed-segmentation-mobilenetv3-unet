"""Small synthetic tests for source-aware dataset preparation.

Run: python -m unittest tests.test_research_data
"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from skimage import io as skio

from scripts.research_data import encode_mask, prepare_dataset
from utils.labels import LABEL_COLORS


class ResearchDataTests(unittest.TestCase):
    def test_rejects_unknown_label_color(self):
        invalid = np.zeros((3, 3, 3), dtype=np.uint8)
        invalid[1, 1] = [9, 8, 7]
        with self.assertRaisesRegex(ValueError, "Unknown ground-truth colors"):
            encode_mask(invalid)

    def test_source_manifest_and_full_validation_patches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "data" / "trainval"
            (raw / "img").mkdir(parents=True)
            (raw / "msk").mkdir(parents=True)
            for num in range(2):
                img = np.full((5, 5, 3), 100 + num, dtype=np.uint8)
                mask = np.empty((5, 5, 3), dtype=np.uint8)
                mask[:] = LABEL_COLORS[0]
                if num == 0:
                    mask[:2, :2] = LABEL_COLORS[2]
                skio.imsave(raw / "img" / f"trainval_{num:02d}.jpg", img, check_contrast=False)
                skio.imsave(raw / "msk" / f"trainval_{num:02d}.png", mask, check_contrast=False)
            stats = prepare_dataset(root, patch_size=4, min_foreground_fraction=0.01)
            self.assertEqual(stats["source_images"], 2)
            self.assertEqual(stats["total_patches"], 8)
            with (raw / "research_patches" / "manifest.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({r["source_image"] for r in rows}, {"trainval_00", "trainval_01"})
            self.assertEqual(len({r["patch_image"] for r in rows}), 8)
            self.assertEqual(sum(int(r["train_eligible"]) for r in rows), 1)
            self.assertEqual(sum(1 for r in rows if r["valid_height"] == "1"), 4)
            for row in rows:
                self.assertTrue((root / row["patch_image"]).exists())
                self.assertTrue((root / row["patch_mask"]).exists())
            with self.assertRaises(FileExistsError):
                prepare_dataset(root, patch_size=4)


if __name__ == "__main__":
    unittest.main()
