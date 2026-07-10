import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from utils.efficiency import FLOP_POLICY, model_size_mb, parameter_counts, summarize_model_efficiency
from utils.model_registry import RESNET_ENCODERS, targets_for_mode
from utils.reporting import generate_prediction_report
from utils.train import resolve_device, set_model


def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate complete evaluation outputs for baseline, proposed, proposed_v2, "
            "full, full_v2, or MobileNetV3 U-Net ablation suites."
        )
    )
    parser.add_argument(
        "mode",
        choices=["baseline", "proposed", "proposed_v2", "full", "full_v2", "ablation"],
        help="Evaluation preset.",
    )
    parser.add_argument(
        "--encoder",
        choices=RESNET_ENCODERS,
        default="resnet34",
        help="ResNet encoder used by baseline models and the baseline U-Net row in ablation mode.",
    )
    parser.add_argument("--root_path", default=".", help="Repository root path.")
    parser.add_argument("--subset", default="test", help="Dataset subset under data/.")
    parser.add_argument(
        "--report_dir",
        default=None,
        help="Output report directory. Defaults to results/reports/<subset>/<mode>_evaluation_<encoder>.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
        help="Benchmark device. Use cuda for RTX 5060 results.",
    )
    parser.add_argument("--input_size", type=int, default=480, help="Benchmark input size, default 480.")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Benchmark warmup iterations.")
    parser.add_argument("--benchmark_iterations", type=int, default=20, help="Timed benchmark iterations.")
    parser.add_argument(
        "--cpu_latency",
        action="store_true",
        help="Also measure CPU latency. This can be slow on larger baselines.",
    )
    parser.add_argument(
        "--skip_efficiency",
        action="store_true",
        help="Skip params/FLOPs/FPS/latency benchmarking and export segmentation metrics only.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use pre0 checkpoint names and instantiate models without ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--allow_missing_checkpoints",
        action="store_true",
        help="Allow efficiency profiling of randomly initialized models when checkpoints are missing.",
    )
    parser.add_argument(
        "--allow_missing_predictions",
        action="store_true",
        help="Skip models whose prediction folders are missing instead of failing.",
    )
    parser.add_argument("--crop_size", type=int, default=400, help="Qualitative report crop size.")
    parser.add_argument("--max_examples", type=int, default=6, help="Qualitative report example count.")
    return parser

def default_report_dir(root_path, subset, mode, encoder):
    if mode == "proposed":
        name = "proposed_evaluation"
    elif mode == "proposed_v2":
        name = "proposed_v2_evaluation"
    elif mode == "full_v2":
        name = f"full_v2_evaluation_{encoder}"
    else:
        name = f"{mode}_evaluation_{encoder}"
    return Path(root_path) / "results" / "reports" / subset / name


def load_checkpoint_state(model, checkpoint_path, device, allow_missing=False):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        if allow_missing:
            return False
        raise FileNotFoundError(
            f"Checkpoint is missing: {checkpoint_path}. "
            "Train first or use --allow_missing_checkpoints for architecture-only efficiency profiling."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return True


def collect_efficiency(args, specs, pretrained):
    if args.skip_efficiency:
        return {}

    device = resolve_device(args.device)
    rows = {}
    for spec in specs:
        checkpoint_path = spec.checkpoint_path(args.root_path, pretrained=pretrained)
        print(f"Benchmarking {spec.model_name} on {device} with {args.input_size}x{args.input_size} input...")
        model = set_model(
            architecture=spec.architecture,
            encoder_name=spec.encoder_name,
            pretrained=False,
            b_bilinear=True,
            replace_stride_with_dilation=False,
            num_classes=3,
        )
        checkpoint_loaded = load_checkpoint_state(
            model,
            checkpoint_path,
            device="cpu",
            allow_missing=args.allow_missing_checkpoints,
        )
        efficiency = {}
        efficiency.update(parameter_counts(model))
        efficiency["model_size_mb"] = float(model_size_mb(model))
        efficiency["efficiency_error"] = ""
        try:
            measured_efficiency = summarize_model_efficiency(
                model,
                input_size=args.input_size,
                device=device,
                warmup_iterations=args.warmup_iterations,
                benchmark_iterations=args.benchmark_iterations,
                use_amp=True,
                include_cpu_latency=args.cpu_latency,
            )
            efficiency.update(measured_efficiency)
        except Exception as exc:  # noqa: BLE001 - keep suite evaluation usable after one benchmark failure.
            efficiency.update(
                {
                    "flops": None,
                    "gflops": None,
                    "flop_policy": FLOP_POLICY,
                    "latency_ms_per_image": None,
                    "fps": None,
                    "peak_gpu_memory_mb": None,
                    "benchmark_device": device,
                    "cpu_latency_ms_per_image": None,
                    "cpu_fps": None,
                    "efficiency_error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"Efficiency benchmark failed for {spec.model_name}: {type(exc).__name__}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        efficiency.update(
            {
                "model": spec.model_name,
                "architecture": spec.architecture,
                "encoder_name": spec.encoder_name,
                "checkpoint": str(checkpoint_path),
                "checkpoint_loaded": checkpoint_loaded,
                "input_size": args.input_size,
            }
        )
        rows[spec.model_name] = efficiency
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def overall_metrics_by_model(manifest):
    result = {}
    for model_report in manifest["models"]:
        metrics = model_report["metrics"]
        result[model_report["model_name"]] = {
            "pixel_accuracy": metrics["pixel_accuracy"],
            "global_accuracy": metrics["global_accuracy"],
            "mean_iou": metrics["mean_iou"],
            "mean_dice": metrics["mean_dice"],
        }
    return result


def build_summary_rows(specs, manifest, efficiency_by_model):
    metrics_by_model = overall_metrics_by_model(manifest)
    rows = []
    for spec in specs:
        if spec.model_name not in metrics_by_model:
            continue
        efficiency = efficiency_by_model.get(spec.model_name, {})
        row = {
            "model": spec.model_name,
            "architecture": spec.architecture,
            "encoder_name": spec.encoder_name,
            "mIoU": metrics_by_model[spec.model_name]["mean_iou"],
            "mean_dice": metrics_by_model[spec.model_name]["mean_dice"],
            "pixel_accuracy": metrics_by_model[spec.model_name]["pixel_accuracy"],
            "trainable_params": efficiency.get("trainable_params"),
            "total_params": efficiency.get("total_params"),
            "benchmark_input_size": efficiency.get("input_size"),
            "gflops": efficiency.get("gflops"),
            "model_size_mb": efficiency.get("model_size_mb"),
            "fps": efficiency.get("fps"),
            "latency_ms_per_image": efficiency.get("latency_ms_per_image"),
            "peak_gpu_memory_mb": efficiency.get("peak_gpu_memory_mb"),
            "cpu_latency_ms_per_image": efficiency.get("cpu_latency_ms_per_image"),
            "efficiency_error": efficiency.get("efficiency_error"),
        }
        rows.append(row)
    return rows


def write_markdown_summary(rows, save_path):
    columns = [
        ("model", "Model"),
        ("mIoU", "mIoU"),
        ("mean_dice", "Mean Dice"),
        ("pixel_accuracy", "Pixel Acc"),
        ("trainable_params", "Trainable Params"),
        ("gflops", "GFLOPs"),
        ("model_size_mb", "Model MB"),
        ("fps", "FPS"),
        ("latency_ms_per_image", "Latency ms/img"),
        ("efficiency_error", "Efficiency Error"),
    ]
    lines = [
        "# Evaluation Summary",
        "",
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Metric notes:",
            "",
            "- Segmentation metrics are pixel-level metrics computed from saved prediction masks versus dataset ground truth.",
            "- GFLOPs use a 480x480 RGB input. Conv/Linear multiply-add operations are counted as 2 FLOPs.",
            "- FPS and latency are single-image inference measurements with batch size 1.",
            "- CPU latency is filled only when `--cpu_latency` is used.",
        ]
    )
    Path(save_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = create_parser().parse_args()
    root_path = Path(args.root_path)
    pretrained = not args.no_pretrained
    specs = targets_for_mode(args.mode, args.encoder)
    report_dir = Path(args.report_dir) if args.report_dir else default_report_dir(root_path, args.subset, args.mode, args.encoder)
    report_dir.mkdir(parents=True, exist_ok=True)

    prediction_entries = []
    evaluated_specs = []
    for spec in specs:
        prediction_dir = spec.prediction_dir(root_path, args.subset)
        if not prediction_dir.is_dir():
            if args.allow_missing_predictions:
                print(f"Skipping {spec.model_name}; missing prediction directory: {prediction_dir}")
                continue
            raise FileNotFoundError(
                f"Prediction directory is missing: {prediction_dir}. "
                "Run prediction first or use --allow_missing_predictions."
            )
        evaluated_specs.append(spec)
        prediction_entries.append(
            {
                "model_name": spec.model_name,
                "prediction_dir": str(prediction_dir),
                "checkpoint": str(spec.checkpoint_path(root_path, pretrained=pretrained)),
            }
        )

    if not prediction_entries:
        raise RuntimeError("No prediction directories were available for evaluation.")

    print(f"Generating segmentation report for {len(prediction_entries)} model(s)...")
    manifest = generate_prediction_report(
        root_path=root_path,
        subset=args.subset,
        prediction_entries=prediction_entries,
        report_dir=report_dir,
        crop_size=args.crop_size,
        max_examples=args.max_examples,
    )

    efficiency_by_model = collect_efficiency(args, evaluated_specs, pretrained=pretrained)
    efficiency_rows = list(efficiency_by_model.values())
    if efficiency_rows:
        pd.DataFrame(efficiency_rows).to_csv(report_dir / "efficiency_metrics.csv", index=False)
        (report_dir / "efficiency_metrics.json").write_text(
            json.dumps(efficiency_rows, indent=2),
            encoding="utf-8",
        )

    summary_rows = build_summary_rows(evaluated_specs, manifest, efficiency_by_model)
    pd.DataFrame(summary_rows).to_csv(report_dir / "evaluation_summary.csv", index=False)
    (report_dir / "evaluation_summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )
    write_markdown_summary(summary_rows, report_dir / "evaluation_summary.md")

    print(f"Evaluation report saved to {report_dir}")


if __name__ == "__main__":
    main()
