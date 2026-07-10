import argparse
from pathlib import Path

from utils.reporting import generate_prediction_report


def parse_prediction_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Prediction input must use MODEL_NAME=prediction_dir, for example "
            "unet_resnet34=results/predictions/test/unet_resnet34"
        )
    model_name, prediction_dir = value.split("=", 1)
    model_name = model_name.strip()
    prediction_dir = prediction_dir.strip()
    if not model_name:
        raise argparse.ArgumentTypeError("MODEL_NAME cannot be empty.")
    if not prediction_dir:
        raise argparse.ArgumentTypeError("prediction_dir cannot be empty.")
    return model_name, prediction_dir


def create_parser():
    parser = argparse.ArgumentParser(
        description="Compare one or more saved prediction directories against dataset ground truth."
    )
    parser.add_argument(
        "subset",
        type=str,
        help="Dataset subset under data/, for example test or test_different_bbch.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction_arg,
        required=True,
        help="Model prediction directory as MODEL_NAME=DIR. Repeat this option to compare models.",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default=".",
        help='Path to root of the project. "data" needs to be subpath of this.',
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help="Directory used to save the comparison report. Defaults to results/reports/<subset>/comparison.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=400,
        help="Crop size in pixels for qualitative report examples.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=6,
        help="Maximum number of representative qualitative examples in the report grid.",
    )
    return parser


def resolve_under_root(root_path, path_text):
    path = Path(path_text)
    return path if path.is_absolute() else Path(root_path) / path


def main():
    args = create_parser().parse_args()
    root_path = Path(args.root_path)
    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else root_path / "results" / "reports" / args.subset / "comparison"
    )

    prediction_entries = []
    for model_name, prediction_dir in args.prediction:
        resolved_dir = resolve_under_root(root_path, prediction_dir)
        prediction_entries.append(
            {
                "model_name": model_name,
                "prediction_dir": str(resolved_dir),
                "checkpoint": "prediction directory only",
            }
        )

    print(f"Comparing {len(prediction_entries)} model prediction set(s) on subset {args.subset}.")
    for entry in prediction_entries:
        print(f"  - {entry['model_name']}: {entry['prediction_dir']}")

    generate_prediction_report(
        root_path=root_path,
        subset=args.subset,
        prediction_entries=prediction_entries,
        report_dir=report_dir,
        crop_size=args.crop_size,
        max_examples=args.max_examples,
    )
    print(f"Comparison report saved to {report_dir}")


if __name__ == "__main__":
    main()
