import unittest
from pathlib import Path

from utils.reporting import normalized_dataset_stem, pair_image_and_mask_paths


class ReportingPairingTest(unittest.TestCase):
    def test_normalized_dataset_stem_strips_common_image_and_mask_suffixes(self):
        self.assertEqual(normalized_dataset_stem(Path("bbch15_img.jpg")), "bbch15")
        self.assertEqual(normalized_dataset_stem(Path("bbch19_msk.png")), "bbch19")
        self.assertEqual(normalized_dataset_stem(Path("test_01.jpg")), "test_01")

    def test_pair_image_and_mask_paths_accepts_bbch_img_msk_suffixes(self):
        images = [Path("bbch19_img.jpg"), Path("bbch15_img.jpg")]
        masks = [Path("bbch15_msk.png"), Path("bbch19_msk.png")]

        paired_images, paired_masks = pair_image_and_mask_paths(images, masks, Path("data/test_different_bbch"))

        self.assertEqual([path.name for path in paired_images], ["bbch15_img.jpg", "bbch19_img.jpg"])
        self.assertEqual([path.name for path in paired_masks], ["bbch15_msk.png", "bbch19_msk.png"])

    def test_pair_image_and_mask_paths_rejects_duplicate_normalized_keys(self):
        images = [Path("bbch15.jpg"), Path("bbch15_img.jpg")]
        masks = [Path("bbch15_msk.png")]

        with self.assertRaisesRegex(ValueError, "normalized stems are duplicated"):
            pair_image_and_mask_paths(images, masks, Path("data/test_different_bbch"))


if __name__ == "__main__":
    unittest.main()
