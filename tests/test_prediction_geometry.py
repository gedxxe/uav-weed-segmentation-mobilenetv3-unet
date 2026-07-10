import unittest

import torch
import torch.nn as nn

from utils.predict import grid_from_mask_shape, predict_one_batch, reshape_predictions_to_images


class PredictionGeometryTest(unittest.TestCase):
    def test_grid_is_derived_from_mask_shape(self):
        self.assertEqual(grid_from_mask_shape((3648, 5472)), (22, 15))
        self.assertEqual(grid_from_mask_shape((2816, 2560)), (10, 11))
        self.assertEqual(grid_from_mask_shape((300, 513)), (3, 2))

    def test_reshape_predictions_supports_non_hardcoded_mask_shape(self):
        mask_shape = (300, 513)
        grid = grid_from_mask_shape(mask_shape)
        slices_per_image = grid[0] * grid[1]
        preds = torch.zeros(slices_per_image * 2, 256, 256, dtype=torch.long)

        images = reshape_predictions_to_images(preds, mask_shape=mask_shape)

        self.assertEqual(len(images), 2)
        self.assertEqual(tuple(images[0].shape), (300, 513, 3))

    def test_predict_one_batch_uses_multiclass_argmax_logits(self):
        class FixedLogitModel(nn.Module):
            def forward(self, x):
                logits = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
                logits[:, 2] = 0.2
                logits[:, 1, 0, 0] = 3.0
                logits[:, 0, 0, 1] = 4.0
                return logits

        inputs = torch.zeros(1, 3, 2, 2)
        targets = torch.zeros(1, 2, 2, dtype=torch.long)

        prediction = predict_one_batch(
            FixedLogitModel(),
            inputs,
            targets,
            device="cpu",
            use_amp=False,
        )

        expected = torch.tensor([[[1, 0], [2, 2]]])
        self.assertTrue(torch.equal(prediction.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
