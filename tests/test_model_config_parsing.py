import unittest
from pathlib import Path

from predict_testset import parse_model_config


class ModelConfigParsingTest(unittest.TestCase):
    def test_parses_existing_resnet_checkpoint_name(self):
        config = parse_model_config(Path("models/unet_resnet34_dil0_bilin1_pre1.pth.tar"))

        self.assertEqual(config["architecture"], "unet")
        self.assertEqual(config["encoder_name"], "resnet34")
        self.assertFalse(config["replace_stride_with_dilation"])
        self.assertTrue(config["b_bilinear"])

    def test_parses_existing_retrained_checkpoint_name(self):
        config = parse_model_config(Path("models/model_unet_resnet34_dil0_bilin1_retrained.pt"))

        self.assertEqual(config["architecture"], "unet")
        self.assertEqual(config["encoder_name"], "resnet34")
        self.assertFalse(config["replace_stride_with_dilation"])
        self.assertTrue(config["b_bilinear"])

    def test_parses_proposed_checkpoint_name_with_underscores(self):
        config = parse_model_config(
            Path("models/unet_mobilenetv3_mobilenetv3_large_dil0_bilin1_pre1.pth.tar")
        )

        self.assertEqual(config["architecture"], "unet_mobilenetv3")
        self.assertEqual(config["encoder_name"], "mobilenetv3_large")
        self.assertFalse(config["replace_stride_with_dilation"])
        self.assertTrue(config["b_bilinear"])

    def test_parses_proposed_ablation_checkpoint_name(self):
        config = parse_model_config(
            Path("models/unet_mobilenetv3_ppm_mobilenetv3_large_dil0_bilin1_pre1.pth.tar")
        )

        self.assertEqual(config["architecture"], "unet_mobilenetv3_ppm")
        self.assertEqual(config["encoder_name"], "mobilenetv3_large")
        self.assertFalse(config["replace_stride_with_dilation"])
        self.assertTrue(config["b_bilinear"])

    def test_parses_proposed_v2_checkpoint_name(self):
        config = parse_model_config(
            Path("models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar")
        )

        self.assertEqual(config["architecture"], "unet_mobilenetv3_aux")
        self.assertEqual(config["encoder_name"], "mobilenetv3_large")
        self.assertFalse(config["replace_stride_with_dilation"])
        self.assertTrue(config["b_bilinear"])


if __name__ == "__main__":
    unittest.main()
