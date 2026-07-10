import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models import mobilenet_v3_large

from utils.model_outputs import FOREGROUND_KEY, SEGMENTATION_KEY

try:
    from torchvision.models import MobileNet_V3_Large_Weights
except ImportError:  # pragma: no cover - compatibility with older torchvision
    MobileNet_V3_Large_Weights = None


def get_torch_checkpoint_dir():
    torch_home = Path(os.environ.get("TORCH_HOME", Path.cwd() / ".cache" / "torch"))
    checkpoint_dir = torch_home / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def initialize_lightweight_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class MobileNetV3LargeEncoder(nn.Module):
    """MobileNetV3-Large feature extractor aligned to the paper's selected stages."""

    feature_indices = (1, 3, 6, 12, 15)
    out_channels = (16, 16, 24, 40, 112, 160)

    def __init__(self, pretrained=True):
        super().__init__()
        weights = None
        if pretrained and MobileNet_V3_Large_Weights is not None:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1

        try:
            backbone = mobilenet_v3_large(weights=None)
        except TypeError:  # pragma: no cover - old torchvision API
            backbone = mobilenet_v3_large(pretrained=False)

        if pretrained:
            if weights is None:
                try:  # pragma: no cover - old torchvision compatibility
                    backbone = mobilenet_v3_large(pretrained=True)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "Could not load pretrained MobileNetV3_Large weights. "
                        "Rerun with --no-pretrained if pretrained download/cache is unavailable."
                    ) from exc
            else:
                try:
                    state_dict = load_state_dict_from_url(
                        weights.url,
                        model_dir=str(get_torch_checkpoint_dir()),
                        progress=True,
                    )
                    backbone.load_state_dict(state_dict)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "Could not load pretrained MobileNetV3_Large weights. "
                        "Rerun with --no-pretrained if pretrained download/cache is unavailable."
                    ) from exc

        last_index = max(self.feature_indices)
        self.high_resolution_stem = nn.Sequential(
            nn.Conv2d(3, self.out_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels[0]),
            nn.Hardswish(inplace=True),
        )
        self.features = nn.ModuleList(list(backbone.features.children())[: last_index + 1])

    def forward(self, x):
        outputs = [self.high_resolution_stem(x)]
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.feature_indices:
                outputs.append(x)
        return outputs


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), branch_channels=None):
        super().__init__()
        if branch_channels is None:
            branch_channels = max(1, in_channels // len(pool_sizes))

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                    nn.Hardswish(inplace=True),
                )
                for pool_size in pool_sizes
            ]
        )
        merged_channels = in_channels + len(pool_sizes) * branch_channels
        self.project = nn.Sequential(
            nn.Conv2d(merged_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        features = [x]
        for branch in self.branches:
            pooled = branch(x)
            features.append(F.interpolate(pooled, size=size, mode="bilinear", align_corners=False))
        return self.project(torch.cat(features, dim=1))


class SqueezeExcitation2d(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        return x * self.block(x)


class MobileNetV3UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, se_reduction=4, use_se=True):
        super().__init__()
        self.skip_channels = skip_channels
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.Hardswish(inplace=True)
        self.se = SqueezeExcitation2d(out_channels, reduction=se_reduction) if use_se else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.Hardswish(inplace=True)

    def forward(self, x, skip=None, output_size=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            if output_size is None:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            else:
                x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.se(x)
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class MobileNetV3UNetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels=MobileNetV3LargeEncoder.out_channels,
        decoder_channels=(160, 112, 64, 32, 16),
        ppm_pool_sizes=(1, 2, 3, 6),
        se_reduction=4,
        use_ppm=True,
        use_se=True,
    ):
        super().__init__()
        if len(encoder_channels) != 6:
            raise ValueError("MobileNetV3UNetDecoder expects six encoder feature stages.")
        if len(decoder_channels) != len(encoder_channels) - 1:
            raise ValueError(
                "MobileNetV3UNetDecoder expects one decoder channel value per skip feature."
            )

        self.ppm_modules = nn.ModuleList(
            [
                PyramidPoolingModule(channels, pool_sizes=ppm_pool_sizes)
                if use_ppm else nn.Identity()
                for channels in encoder_channels[:-1]
            ]
        )

        skip_channels = list(encoder_channels[:-1])[::-1]
        in_channels = encoder_channels[-1]
        blocks = []
        for skip_ch, out_ch in zip(skip_channels, decoder_channels):
            blocks.append(
                MobileNetV3UNetDecoderBlock(
                    in_channels=in_channels,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    se_reduction=se_reduction,
                    use_se=use_se,
                )
            )
            in_channels = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = decoder_channels[-1]

    def forward(self, features, output_size):
        if len(features) != 6:
            raise ValueError(f"Expected six MobileNetV3 feature maps, got {len(features)}.")

        ppm_skips = [ppm(feature) for ppm, feature in zip(self.ppm_modules, features[:-1])]
        x = features[-1]

        for block, skip in zip(self.blocks, reversed(ppm_skips)):
            x = block(x, skip=skip)
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class MobileNetV3UNet(nn.Module):
    """
    Proposed lightweight U-Net variant:
    MobileNetV3-Large encoder, PPM-enhanced skip features, and SE/H-swish decoder.
    """

    def __init__(
        self,
        encoder_name="mobilenetv3_large",
        num_classes=3,
        pretrained=True,
        decoder_channels=(160, 112, 64, 32, 16),
        ppm_pool_sizes=(1, 2, 3, 6),
        se_reduction=4,
        use_ppm=True,
        use_se=True,
        use_auxiliary_foreground_head=False,
        foreground_classes=2,
    ):
        super().__init__()
        if encoder_name != "mobilenetv3_large":
            raise ValueError(
                "MobileNetV3UNet supports only encoder_name='mobilenetv3_large'. "
                f"Got {encoder_name!r}."
            )

        print(f"{encoder_name=}")
        if pretrained:
            print("LOADING MOBILENETV3_LARGE PRETRAINED WEIGHTS FROM IMAGENET")
        else:
            print("INITIALIZING MOBILENETV3_LARGE BACKBONE WITHOUT IMAGENET PRETRAINED WEIGHTS")

        self.backbone = MobileNetV3LargeEncoder(pretrained=pretrained)
        self.use_auxiliary_foreground_head = use_auxiliary_foreground_head
        self.decoder = MobileNetV3UNetDecoder(
            encoder_channels=MobileNetV3LargeEncoder.out_channels,
            decoder_channels=decoder_channels,
            ppm_pool_sizes=ppm_pool_sizes,
            se_reduction=se_reduction,
            use_ppm=use_ppm,
            use_se=use_se,
        )
        self.segmentation_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(self.decoder.out_channels, num_classes, kernel_size=3, padding=1),
        )
        self.foreground_head = (
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(self.decoder.out_channels, foreground_classes, kernel_size=3, padding=1),
            )
            if use_auxiliary_foreground_head
            else None
        )
        self.initialize()

    def initialize(self):
        initialize_lightweight_decoder(self.backbone.high_resolution_stem)
        initialize_lightweight_decoder(self.decoder)
        initialize_lightweight_decoder(self.segmentation_head)
        if self.foreground_head is not None:
            initialize_lightweight_decoder(self.foreground_head)

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.backbone(x)
        decoder_output = self.decoder(features, output_size=input_size)
        masks = self.segmentation_head(decoder_output)
        if masks.shape[-2:] != input_size:
            masks = F.interpolate(masks, size=input_size, mode="bilinear", align_corners=False)
        if self.foreground_head is None:
            return masks

        foreground = self.foreground_head(decoder_output)
        if foreground.shape[-2:] != input_size:
            foreground = F.interpolate(foreground, size=input_size, mode="bilinear", align_corners=False)
        return {
            SEGMENTATION_KEY: masks,
            FOREGROUND_KEY: foreground,
        }
