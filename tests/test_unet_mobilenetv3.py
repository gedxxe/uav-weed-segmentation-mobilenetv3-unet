import unittest

import torch

from utils.manual_unet_mobilenetv3 import MobileNetV3UNet
from utils.model_outputs import FOREGROUND_KEY, SEGMENTATION_KEY
from utils.train import set_model


class MobileNetV3UNetSmokeTest(unittest.TestCase):
    def test_forward_preserves_spatial_shape(self):
        model = MobileNetV3UNet(pretrained=False, num_classes=3)
        model.eval()

        x = torch.randn(1, 3, 129, 157)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 3, 129, 157))

    def test_parameter_budget_stays_lightweight(self):
        model = MobileNetV3UNet(pretrained=False, num_classes=3)
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

        self.assertLess(trainable_params, 8_000_000)
        self.assertGreater(trainable_params, 1_000_000)

    def test_train_factory_builds_proposed_model(self):
        model = set_model(
            architecture="unet_mobilenetv3",
            encoder_name="mobilenetv3_large",
            pretrained=False,
            b_bilinear=True,
            replace_stride_with_dilation=False,
            num_classes=3,
        )

        self.assertIsInstance(model, MobileNetV3UNet)

    def test_train_factory_builds_ablation_variants(self):
        for architecture in ("unet_mobilenetv3_base", "unet_mobilenetv3_ppm"):
            model = set_model(
                architecture=architecture,
                encoder_name="mobilenetv3_large",
                pretrained=False,
                b_bilinear=True,
                replace_stride_with_dilation=False,
                num_classes=3,
            )
            model.eval()
            x = torch.randn(1, 3, 65, 73)
            with torch.no_grad():
                y = model(x)
            self.assertEqual(tuple(y.shape), (1, 3, 65, 73))

    def test_auxiliary_variant_returns_segmentation_and_foreground_logits(self):
        model = set_model(
            architecture="unet_mobilenetv3_aux",
            encoder_name="mobilenetv3_large",
            pretrained=False,
            b_bilinear=True,
            replace_stride_with_dilation=False,
            num_classes=3,
        )
        model.eval()

        x = torch.randn(1, 3, 65, 73)
        with torch.no_grad():
            y = model(x)

        self.assertIn(SEGMENTATION_KEY, y)
        self.assertIn(FOREGROUND_KEY, y)
        self.assertEqual(tuple(y[SEGMENTATION_KEY].shape), (1, 3, 65, 73))
        self.assertEqual(tuple(y[FOREGROUND_KEY].shape), (1, 2, 65, 73))


if __name__ == "__main__":
    unittest.main()
