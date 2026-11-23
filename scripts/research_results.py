"""Summarize frozen-model held-out segmentation reports in a single CSV table.

This script reads existing reports only; it never trains or tunes on test data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def compare(root, run_ids, subset="test"):
    root = Path(root).resolve()
    output = root / "results" / "research_runs" / "_comparison"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    capture_rows = []
    for run_id in run_ids:
        report = root / "results" / "research_runs" / run_id / "reports" / subset
        metrics_path = report / "metrics_summary.json"
        per_image = report / "metrics_per_image.csv"
        if not metrics_path.is_file() or not per_image.is_file():
            raise FileNotFoundError(
                f"Missing report for {run_id} / {subset}: {metrics_path} or {per_image}"
            )
        models = json.loads(metrics_path.read_text(encoding="utf-8"))
        if len(models) != 1:
            raise ValueError(f"Expected one model per report, got {list(models)} in {report}")
        model_name, metrics = next(iter(models.items()))
        class_metrics = {c["class"].lower(): c for c in metrics["per_class"]}
        if not {"background", "sorghum", "weed"}.issubset(class_metrics):
            raise ValueError(f"Expected Background, Sorghum and Weed in {metrics_path}")
        weed = class_metrics["weed"]
        records.append({
            "run_id": run_id, "model": model_name, "subset": subset,
            "pixel_accuracy": metrics["pixel_accuracy"],
            "miou": metrics["mean_iou"], "mean_dice": metrics["mean_dice"],
            "background_iou": class_metrics["background"]["iou"],
            "sorghum_iou": class_metrics["sorghum"]["iou"],
            "weed_iou": weed["iou"],
            "weed_precision": weed["precision"],
            "weed_recall": weed["recall"],
            "weed_f1": weed["f1_score"],
            "n_labeled_pixels": metrics["support"],
        })
        with per_image.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                capture_rows.append({
                    "run_id": run_id, "model": model_name,
                    "subset": subset, "image": row["image"],
                    "pixel_accuracy": row["pixel_accuracy"],
                    "miou": row["mean_iou"], "mean_dice": row["mean_dice"],
                    "weed_iou": row.get("iou_weed", ""),
                    "weed_recall": row.get("recall_weed", ""),
                })
    path = output / f"{subset}_models.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    per_path = output / f"{subset}_per_image.csv"
    with per_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(capture_rows[0]))
        writer.writeheader()
        writer.writerows(capture_rows)
    return path, per_path, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--subset", choices=("test", "test_different_bbch"), default="test")
    args = parser.parse_args()
    a, b, rows = compare(args.root, args.runs, args.subset)
    print("Comparison:", a)
    print("Per-image:", b)
    for row in rows:
        print(
            f"{row['run_id']}: mIoU={row['miou']:.4f}, "
            f"WeedIoU={row['weed_iou']:.4f}, WeedRecall={row['weed_recall']:.4f}"
        )


if __name__ == "__main__":
    main()
