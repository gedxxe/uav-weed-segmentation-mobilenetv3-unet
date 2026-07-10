from dataclasses import dataclass
from pathlib import Path


BASELINE_ARCHITECTURES = ["fcn8s", "fcn16s", "fcn32s", "unet", "dlplus"]
PROPOSED_ARCHITECTURES = [
    "unet_mobilenetv3_base",
    "unet_mobilenetv3_ppm",
    "unet_mobilenetv3",
    "unet_mobilenetv3_aux",
]
ARCHITECTURE_CHOICES = BASELINE_ARCHITECTURES + PROPOSED_ARCHITECTURES

RESNET_ENCODERS = ["resnet18", "resnet34", "resnet50", "resnet101"]
PROPOSED_ENCODERS = ["mobilenetv3_large"]
ENCODER_CHOICES = RESNET_ENCODERS + PROPOSED_ENCODERS

PROPOSED_ARCHITECTURE = "unet_mobilenetv3"
PROPOSED_V2_ARCHITECTURE = "unet_mobilenetv3_aux"
PROPOSED_ENCODER = "mobilenetv3_large"
PROPOSED_MODEL_NAME = "unet_mobilenetv3"
PROPOSED_V2_MODEL_NAME = "unet_mobilenetv3_aux"
ABLATION_ARCHITECTURES = [
    "unet_mobilenetv3_base",
    "unet_mobilenetv3_ppm",
    "unet_mobilenetv3",
    "unet_mobilenetv3_aux",
]

DEFAULT_BATCH_BY_ENCODER = {
    "resnet18": 8,
    "resnet34": 8,
    "resnet50": 4,
    "resnet101": 2,
}


@dataclass(frozen=True)
class ModelTarget:
    architecture: str
    encoder_name: str
    model_name: str

    def checkpoint_path(self, root_path, pretrained=True):
        suffix = "pre1" if pretrained else "pre0"
        return (
            Path(root_path)
            / "models"
            / f"{self.architecture}_{self.encoder_name}_dil0_bilin1_{suffix}.pth.tar"
        )

    def prediction_dir(self, root_path, subset):
        return Path(root_path) / "results" / "predictions" / subset / self.model_name

    def report_dir(self, root_path, subset):
        return Path(root_path) / "results" / "reports" / subset / self.model_name


def baseline_targets(encoder):
    return [
        ModelTarget(
            architecture=architecture,
            encoder_name=encoder,
            model_name=f"{architecture}_{encoder}",
        )
        for architecture in BASELINE_ARCHITECTURES
    ]


def proposed_target():
    return ModelTarget(
        architecture=PROPOSED_ARCHITECTURE,
        encoder_name=PROPOSED_ENCODER,
        model_name=PROPOSED_MODEL_NAME,
    )


def proposed_v2_target():
    return ModelTarget(
        architecture=PROPOSED_V2_ARCHITECTURE,
        encoder_name=PROPOSED_ENCODER,
        model_name=PROPOSED_V2_MODEL_NAME,
    )


def full_targets(encoder):
    return baseline_targets(encoder) + [proposed_target()]


def full_v2_targets(encoder):
    return baseline_targets(encoder) + [proposed_v2_target()]


def ablation_targets(encoder):
    return [ModelTarget("unet", encoder, f"unet_{encoder}")] + [
        ModelTarget(architecture, PROPOSED_ENCODER, architecture)
        for architecture in ABLATION_ARCHITECTURES
    ]


def targets_for_mode(mode, encoder):
    if mode == "baseline":
        return baseline_targets(encoder)
    if mode == "proposed":
        return [proposed_target()]
    if mode == "proposed_v2":
        return [proposed_v2_target()]
    if mode == "full":
        return full_targets(encoder)
    if mode == "full_v2":
        return full_v2_targets(encoder)
    if mode == "ablation":
        return ablation_targets(encoder)
    raise ValueError(f"Unsupported model suite mode: {mode}")
