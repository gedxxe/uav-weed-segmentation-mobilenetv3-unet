import unittest

import torch

from utils.losses import create_loss
from utils.model_outputs import FOREGROUND_KEY, SEGMENTATION_KEY


class AuxiliaryForegroundLossTest(unittest.TestCase):
    def test_auxiliary_foreground_loss_accepts_structured_output(self):
        loss_fn = create_loss(
            "ce_dice_aux_foreground",
            foreground_aux_weight=0.3,
        )
        predictions = {
            SEGMENTATION_KEY: torch.randn(2, 3, 8, 8, requires_grad=True),
            FOREGROUND_KEY: torch.randn(2, 2, 8, 8, requires_grad=True),
        }
        targets = torch.randint(0, 3, (2, 8, 8))

        loss = loss_fn(predictions, targets)

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(predictions[SEGMENTATION_KEY].grad)
        self.assertIsNotNone(predictions[FOREGROUND_KEY].grad)


if __name__ == "__main__":
    unittest.main()
