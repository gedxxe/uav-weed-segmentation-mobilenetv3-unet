import unittest

import numpy as np

from utils.segmentation_metrics import confusion_matrix_from_arrays, metrics_from_confusion


class SegmentationMetricsTest(unittest.TestCase):
    def test_metrics_from_confusion_include_iou_and_dice(self):
        gt = np.array([[0, 1], [2, 2]], dtype=np.uint8)
        pred = np.array([[0, 2], [2, 1]], dtype=np.uint8)

        confusion = confusion_matrix_from_arrays([gt], [pred], num_classes=3)
        metrics = metrics_from_confusion(confusion, class_names=["Background", "Sorghum", "Weed"])

        expected_confusion = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
            ]
        )
        self.assertTrue(np.array_equal(confusion, expected_confusion))
        self.assertAlmostEqual(metrics["pixel_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["per_class"][0]["iou"], 1.0)
        self.assertAlmostEqual(metrics["per_class"][1]["dice"], 0.0)
        self.assertAlmostEqual(metrics["per_class"][2]["dice"], 0.5)


if __name__ == "__main__":
    unittest.main()
