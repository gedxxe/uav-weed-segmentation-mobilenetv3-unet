import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.model_registry import (
    ABLATION_ARCHITECTURES,
    BASELINE_ARCHITECTURES,
    DEFAULT_BATCH_BY_ENCODER,
    PROPOSED_ARCHITECTURE,
    PROPOSED_ENCODER,
    PROPOSED_MODEL_NAME,
    PROPOSED_V2_ARCHITECTURE,
    PROPOSED_V2_MODEL_NAME,
    RESNET_ENCODERS,
)

CLASS_WEIGHT_STRATEGIES = ["inverse_frequency", "sqrt_inverse"]
VALIDATION_LOSSES = ["dice", "same", "macro_f1", "weed_f1", "foreground_macro_f1"]
LOSS_CHOICES = ["dice", "ce", "ce_dice", "ce_dice_aux_foreground"]


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    encoder_name: str
    model_name: str
    batch_size: int
    is_proposed: bool = False
    loss_name: str = "dice"
    class_weights: str = "none"
    class_weight_strategy: str = "inverse_frequency"
    validation_loss: str = "dice"
    foreground_aux_weight: float = 0.0

    def checkpoint_path(self, root_path, pretrained):
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


def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline-only, proposed-only, or full six-model training/prediction/comparison suites. "
            "Use --plan_only before long runs."
        )
    )
    parser.add_argument(
        "mode",
        choices=["baseline", "proposed", "proposed_v2", "full", "full_v2", "ablation"],
        help=(
            "baseline = five ResNet-backed baselines; proposed = MobileNetV3 U-Net only; "
            "proposed_v2 = MobileNetV3 U-Net with auxiliary foreground head; "
            "full = baseline + proposed; full_v2 = baseline + proposed_v2; "
            "ablation = U-Net baseline + MobileNetV3 U-Net variants."
        ),
    )
    parser.add_argument(
        "--encoder",
        choices=RESNET_ENCODERS,
        default="resnet34",
        help="ResNet encoder used for baseline models.",
    )
    parser.add_argument("--root_path", default=".", help="Repository root path.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for baseline models. Defaults to an encoder-specific RTX 5060-safe value.",
    )
    parser.add_argument(
        "--proposed_batch_size",
        type=int,
        default=None,
        help="Batch size for proposed model. Defaults to --batch_size, or the encoder-specific default.",
    )
    parser.add_argument("--num_workers", type=int, default=2, help="Training DataLoader workers.")
    parser.add_argument("--n_folds", type=int, default=2, help="Cross-validation fold count.")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum epochs per fold.")
    parser.add_argument("--n_trials", type=int, default=1, help="Optuna trial count.")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Epochs without validation improvement before early stopping.")
    parser.add_argument("--lr_scheduler_patience", type=int, default=5, help="ReduceLROnPlateau patience in epochs.")
    parser.add_argument("--subset", default="test", help="Dataset subset for prediction/reporting.")
    parser.add_argument("--run_prefix", default=None, help="Experiment DB prefix. Defaults to suite_<mode>_<encoder>.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda", help="Training/prediction device.")
    parser.add_argument(
        "--loss",
        choices=LOSS_CHOICES,
        default="dice",
        help="Loss for baseline models.",
    )
    parser.add_argument(
        "--class_weights",
        choices=["none", "auto"],
        default="none",
        help="Class weighting mode for baseline models.",
    )
    parser.add_argument(
        "--proposed_loss",
        choices=LOSS_CHOICES,
        default="ce_dice",
        help="Loss for the proposed MobileNetV3 U-Net.",
    )
    parser.add_argument(
        "--proposed_class_weights",
        choices=["none", "auto"],
        default="auto",
        help="Class weighting mode for the proposed MobileNetV3 U-Net.",
    )
    parser.add_argument("--class_weight_max", type=float, default=5.0, help="Maximum auto class-weight clip value.")
    parser.add_argument(
        "--class_weight_strategy",
        choices=CLASS_WEIGHT_STRATEGIES,
        default="inverse_frequency",
        help="Class-weight formula for baseline models when --class_weights auto is used.",
    )
    parser.add_argument(
        "--proposed_class_weight_strategy",
        choices=CLASS_WEIGHT_STRATEGIES,
        default="inverse_frequency",
        help="Class-weight formula for the proposed MobileNetV3 U-Net when --proposed_class_weights auto is used.",
    )
    parser.add_argument("--ce_weight", type=float, default=1.0, help="CE term weight for ce_dice loss.")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="Dice term weight for ce_dice loss.")
    parser.add_argument(
        "--foreground_aux_weight",
        type=float,
        default=0.3,
        help="Auxiliary BG-vs-vegetation CE term weight for proposed_v2.",
    )
    parser.add_argument(
        "--validation_loss",
        choices=VALIDATION_LOSSES,
        default="dice",
        help="Checkpoint selection loss for baseline models.",
    )
    parser.add_argument(
        "--proposed_validation_loss",
        choices=VALIDATION_LOSSES,
        default="macro_f1",
        help="Checkpoint selection loss for the proposed MobileNetV3 U-Net.",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    parser.add_argument("--clean_study", action="store_true", help="Delete existing Optuna study before training.")
    parser.add_argument("--skip_training", action="store_true", help="Skip all training and use existing checkpoints.")
    parser.add_argument("--skip_prediction", action="store_true", help="Skip prediction and comparison.")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip complete evaluation summary generation.")
    parser.add_argument("--skip_efficiency", action="store_true", help="Skip efficiency benchmarking in evaluation.")
    parser.add_argument("--evaluation_input_size", type=int, default=480, help="Input size for GFLOPs/latency evaluation.")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Evaluation benchmark warmup iterations.")
    parser.add_argument("--benchmark_iterations", type=int, default=20, help="Evaluation benchmark timed iterations.")
    parser.add_argument("--cpu_latency", action="store_true", help="Also benchmark CPU inference latency during evaluation.")
    parser.add_argument(
        "--skip_proposed_training",
        action="store_true",
        help="In full mode, do not retrain proposed model but still predict/report it from an existing checkpoint.",
    )
    parser.add_argument("--plan_only", action="store_true", help="Print commands without executing them.")
    return parser


def build_model_specs(args):
    baseline_batch_size = args.batch_size or DEFAULT_BATCH_BY_ENCODER[args.encoder]
    proposed_batch_size = args.proposed_batch_size or baseline_batch_size
    proposed_loss = args.proposed_loss
    if args.mode in {"proposed_v2", "full_v2"} and proposed_loss == "ce_dice":
        proposed_loss = "ce_dice_aux_foreground"
    models = []

    if args.mode in {"baseline", "full", "full_v2"}:
        models.extend(
            ModelSpec(
                architecture=architecture,
                encoder_name=args.encoder,
                model_name=f"{architecture}_{args.encoder}",
                batch_size=baseline_batch_size,
                loss_name=args.loss,
                class_weights=args.class_weights,
                class_weight_strategy=args.class_weight_strategy,
                validation_loss=args.validation_loss,
            )
            for architecture in BASELINE_ARCHITECTURES
        )

    if args.mode == "ablation":
        models.append(
            ModelSpec(
                architecture="unet",
                encoder_name=args.encoder,
                model_name=f"unet_{args.encoder}",
                batch_size=baseline_batch_size,
                loss_name=args.loss,
                class_weights=args.class_weights,
                class_weight_strategy=args.class_weight_strategy,
                validation_loss=args.validation_loss,
            )
        )
        for architecture in ABLATION_ARCHITECTURES:
            architecture_loss = proposed_loss
            if architecture == PROPOSED_V2_ARCHITECTURE and args.proposed_loss == "ce_dice":
                architecture_loss = "ce_dice_aux_foreground"
            models.append(
                ModelSpec(
                    architecture=architecture,
                    encoder_name=PROPOSED_ENCODER,
                    model_name=architecture,
                    batch_size=proposed_batch_size,
                    is_proposed=architecture in {PROPOSED_ARCHITECTURE, PROPOSED_V2_ARCHITECTURE},
                    loss_name=architecture_loss,
                    class_weights=args.proposed_class_weights,
                    class_weight_strategy=args.proposed_class_weight_strategy,
                    validation_loss=args.proposed_validation_loss,
                )
            )

    if args.mode in {"proposed", "full"}:
        models.append(
            ModelSpec(
                architecture=PROPOSED_ARCHITECTURE,
                encoder_name=PROPOSED_ENCODER,
                model_name=PROPOSED_MODEL_NAME,
                batch_size=proposed_batch_size,
                is_proposed=True,
                loss_name=proposed_loss,
                class_weights=args.proposed_class_weights,
                class_weight_strategy=args.proposed_class_weight_strategy,
                validation_loss=args.proposed_validation_loss,
            )
        )

    if args.mode in {"proposed_v2", "full_v2"}:
        models.append(
            ModelSpec(
                architecture=PROPOSED_V2_ARCHITECTURE,
                encoder_name=PROPOSED_ENCODER,
                model_name=PROPOSED_V2_MODEL_NAME,
                batch_size=proposed_batch_size,
                is_proposed=True,
                loss_name=proposed_loss,
                class_weights=args.proposed_class_weights,
                class_weight_strategy=args.proposed_class_weight_strategy,
                validation_loss=args.proposed_validation_loss,
                foreground_aux_weight=args.foreground_aux_weight,
            )
        )

    return models


def printable_command(command):
    return " ".join(str(part) for part in command)


def run_step(title, command, plan_only):
    print()
    print(f"== {title} ==")
    print(printable_command(command))
    if plan_only:
        return
    subprocess.run(command, check=True)


def train_command(args, spec, run_prefix):
    command = [
        sys.executable,
        "train.py",
        spec.architecture,
        spec.encoder_name,
        "--root_path",
        args.root_path,
        "--device",
        args.device,
        "--batch_size",
        str(spec.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--n_folds",
        str(args.n_folds),
        "--n_trials",
        str(args.n_trials),
        "--max_epochs",
        str(args.max_epochs),
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--lr_scheduler_patience",
        str(args.lr_scheduler_patience),
        "--run_prefix",
        run_prefix,
        "--loss",
        spec.loss_name,
        "--class_weights",
        spec.class_weights,
        "--class_weight_max",
        str(args.class_weight_max),
        "--class_weight_strategy",
        spec.class_weight_strategy,
        "--ce_weight",
        str(args.ce_weight),
        "--dice_weight",
        str(args.dice_weight),
        "--foreground_aux_weight",
        str(spec.foreground_aux_weight),
        "--validation_loss",
        spec.validation_loss,
        "--save_checkpoint",
    ]
    if args.no_pretrained:
        command.append("--no-pretrained")
    if args.clean_study:
        command.append("--b_clean_study")
    return command


def predict_command(args, spec, checkpoint):
    return [
        sys.executable,
        "predict_testset.py",
        str(checkpoint),
        args.subset,
        "--root_path",
        args.root_path,
        "--device",
        args.device,
        "--num_workers",
        "0",
        "--model_name",
        spec.model_name,
        "--output_dir",
        str(spec.prediction_dir(args.root_path, args.subset)),
        "--report_dir",
        str(spec.report_dir(args.root_path, args.subset)),
    ]


def compare_command(args, models):
    if args.mode == "proposed":
        report_name = "proposed_model"
    elif args.mode == "proposed_v2":
        report_name = "proposed_v2_model"
    elif args.mode == "baseline":
        report_name = f"baseline_comparison_{args.encoder}"
    elif args.mode == "ablation":
        report_name = f"ablation_mobilenetv3_unet_{args.encoder}"
    elif args.mode == "full_v2":
        report_name = f"architecture_comparison_{args.encoder}_proposed_v2"
    else:
        report_name = f"architecture_comparison_{args.encoder}"

    command = [
        sys.executable,
        "compare_model_predictions.py",
        args.subset,
        "--root_path",
        args.root_path,
    ]
    for spec in models:
        command.extend(
            [
                "--prediction",
                f"{spec.model_name}={spec.prediction_dir(args.root_path, args.subset)}",
            ]
        )
    command.extend(
        [
            "--report_dir",
            str(Path(args.root_path) / "results" / "reports" / args.subset / report_name),
        ]
    )
    return command


def evaluation_command(args):
    command = [
        sys.executable,
        "evaluate_model_suite.py",
        args.mode,
        "--root_path",
        args.root_path,
        "--subset",
        args.subset,
        "--device",
        args.device,
        "--input_size",
        str(args.evaluation_input_size),
        "--warmup_iterations",
        str(args.warmup_iterations),
        "--benchmark_iterations",
        str(args.benchmark_iterations),
    ]
    if args.mode not in {"proposed", "proposed_v2"}:
        command.extend(["--encoder", args.encoder])
    if args.no_pretrained:
        command.append("--no-pretrained")
    if args.skip_efficiency:
        command.append("--skip_efficiency")
    if args.cpu_latency:
        command.append("--cpu_latency")
    return command


def main():
    args = create_parser().parse_args()
    models = build_model_specs(args)
    pretrained = not args.no_pretrained
    run_prefix = args.run_prefix or f"suite_{args.mode}_{args.encoder}"

    print(f"Mode: {args.mode}")
    if args.mode in {"baseline", "full", "full_v2", "ablation"}:
        print(f"Baseline encoder: {args.encoder}")
    else:
        print("Baseline encoder: not used")
    print(f"Models: {', '.join(spec.model_name for spec in models)}")

    for spec in models:
        checkpoint = spec.checkpoint_path(args.root_path, pretrained)
        should_train = not args.skip_training
        if spec.is_proposed and args.skip_proposed_training:
            should_train = False

        if should_train:
            run_step(
                title=f"Train {spec.model_name}",
                command=train_command(args, spec, run_prefix),
                plan_only=args.plan_only,
            )

        if not args.skip_prediction:
            if not args.plan_only and not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Expected checkpoint was not found: {checkpoint}. "
                    "Train first or rerun without --skip_training/--skip_proposed_training."
                )
            run_step(
                title=f"Predict and report {spec.model_name}",
                command=predict_command(args, spec, checkpoint),
                plan_only=args.plan_only,
            )

    if not args.skip_prediction:
        run_step(
            title=f"Build {len(models)}-model {args.mode} comparison report",
            command=compare_command(args, models),
            plan_only=args.plan_only,
        )
        if not args.skip_evaluation:
            run_step(
                title=f"Build complete {args.mode} evaluation summary",
                command=evaluation_command(args),
                plan_only=args.plan_only,
            )

    if args.plan_only:
        print()
        print("Plan only. No training, prediction, or comparison was executed.")


if __name__ == "__main__":
    main()
