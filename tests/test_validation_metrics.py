import unittest

import torch

from utils.train import confusion_matrix_from_logits, f1_loss_from_confusion


class ValidationMetricTest(unittest.TestCase):
    def test_confusion_matrix_from_logits_uses_true_rows_predicted_columns(self):
        targets = torch.tensor([[[0, 1], [2, 2]]])
        predicted = torch.tensor([[[0, 2], [2, 1]]])
        logits = torch.zeros(1, 3, 2, 2)
        logits.scatter_(1, predicted.unsqueeze(1), 10.0)

        confusion = confusion_matrix_from_logits(logits, targets, num_classes=3)

        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        self.assertTrue(torch.equal(confusion.cpu(), expected))

    def test_f1_losses_match_manual_values(self):
        confusion = torch.tensor(
            [
                [5.0, 0.0, 0.0],
                [0.0, 3.0, 1.0],
                [0.0, 2.0, 2.0],
            ]
        )

        macro_loss = f1_loss_from_confusion(confusion, average="macro")
        weed_loss = f1_loss_from_confusion(confusion, average="class", class_index=2)
        foreground_loss = f1_loss_from_confusion(
            confusion,
            average="selected_macro",
            class_indices=(1, 2),
        )

        self.assertAlmostEqual(macro_loss.item(), 0.25396824, places=6)
        self.assertAlmostEqual(weed_loss.item(), 0.42857146, places=6)
        self.assertAlmostEqual(foreground_loss.item(), 0.38095242, places=6)


if __name__ == "__main__":
    unittest.main()
