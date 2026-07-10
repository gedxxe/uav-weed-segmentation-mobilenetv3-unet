import json
import os
from pathlib import Path

_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from utils.labels import CLASS_LABELS, CLASS_NAMES, LABEL_COLORS
from utils.patch_utils import load_image, load_mask
from utils.predict import convert_labelmap_to_color
from utils.segmentation_metrics import (
    confusion_matrix_from_arrays,
    flatten_metrics_for_summary,
    metrics_from_confusion,
)

DIFF_COLORS = {
    (0, 0): (0, 0, 0),
    (0, 1): (39, 190, 205),
    (0, 2): (151, 91, 75),
    (1, 0): (150, 103, 190),
    (1, 1): (0, 0, 0),
    (1, 2): (220, 38, 38),
    (2, 0): (214, 101, 178),
    (2, 1): (192, 200, 25),
    (2, 2): (0, 0, 0),
}
ERROR_COLORS = {
    "correct": (0, 0, 0),
    "false_positive_foreground": (220, 38, 38),
    "false_negative_foreground": (37, 99, 235),
    "wrong_foreground_class": (234, 179, 8),
}


def model_id_from_path(path):
    name = Path(path).name
    for suffix in (".pth.tar", ".pt", ".pth"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(path).stem


def prediction_path_for_image(prediction_dir, image_path):
    return Path(prediction_dir) / f"{Path(image_path).stem}_pred.png"


def normalized_dataset_stem(path):
    stem = Path(path).stem
    lowered = stem.casefold()
    for suffix in ("_img", "_image", "_msk", "_mask"):
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _paths_by_normalized_stem(paths, role, data_path):
    mapping = {}
    duplicates = {}
    for path in paths:
        key = normalized_dataset_stem(path)
        if key in mapping:
            duplicates.setdefault(key, [mapping[key]]).append(path)
        else:
            mapping[key] = path
    if duplicates:
        duplicate_text = "; ".join(
            f"{key}: {[item.name for item in duplicate_paths]}"
            for key, duplicate_paths in sorted(duplicates.items())
        )
        raise ValueError(
            f"Ambiguous {role} names in {data_path}; normalized stems are duplicated: {duplicate_text}"
        )
    return mapping


def pair_image_and_mask_paths(image_paths, gt_paths, data_path):
    image_paths = sorted(image_paths)
    gt_paths = sorted(gt_paths)
    if [path.stem for path in image_paths] == [path.stem for path in gt_paths]:
        return image_paths, gt_paths

    image_by_key = _paths_by_normalized_stem(image_paths, "image", data_path)
    gt_by_key = _paths_by_normalized_stem(gt_paths, "ground-truth mask", data_path)
    image_keys = set(image_by_key)
    gt_keys = set(gt_by_key)
    if image_keys != gt_keys:
        raise ValueError(
            f"Image and ground-truth mask names do not match in {data_path}. "
            f"Image stems: {[path.stem for path in image_paths]}; "
            f"mask stems: {[path.stem for path in gt_paths]}; "
            f"normalized image keys: {sorted(image_keys)}; "
            f"normalized mask keys: {sorted(gt_keys)}"
        )

    paired_keys = [normalized_dataset_stem(path) for path in image_paths]
    paired_gt_paths = [gt_by_key[key] for key in paired_keys]
    return image_paths, paired_gt_paths


def load_prediction_set(root_path, subset, prediction_dir):
    data_path = Path(root_path) / "data" / subset
    image_paths = sorted((data_path / "img").glob("*.jpg"))
    gt_paths = sorted((data_path / "msk").glob("*.png"))
    image_paths, gt_paths = pair_image_and_mask_paths(image_paths, gt_paths, data_path)

    prediction_paths = [prediction_path_for_image(prediction_dir, path) for path in image_paths]

    missing = [path for path in prediction_paths if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing prediction files:\n{missing_text}")

    images = [load_image(path) for path in image_paths]
    gts = [load_mask(path) for path in gt_paths]
    preds = [load_mask(path) for path in prediction_paths]
    return image_paths, gt_paths, prediction_paths, images, gts, preds


def difference_map(gt, pred):
    diff = np.zeros((*gt.shape, 3), dtype=np.uint8)
    for pair, color in DIFF_COLORS.items():
        diff[(gt == pair[0]) & (pred == pair[1])] = color
    return diff


def overlay_prediction(image, pred, alpha=0.45):
    overlay = image.astype(np.float32).copy()
    pred_color = convert_labelmap_to_color(pred, LABEL_COLORS).astype(np.float32)
    foreground = pred > 0
    overlay[foreground] = (
        (1.0 - alpha) * overlay[foreground]
        + alpha * pred_color[foreground]
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def error_map(gt, pred):
    errors = np.zeros((*gt.shape, 3), dtype=np.uint8)
    false_positive = (gt == 0) & (pred > 0)
    false_negative = (gt > 0) & (pred == 0)
    wrong_foreground_class = (gt > 0) & (pred > 0) & (gt != pred)
    errors[false_positive] = ERROR_COLORS["false_positive_foreground"]
    errors[false_negative] = ERROR_COLORS["false_negative_foreground"]
    errors[wrong_foreground_class] = ERROR_COLORS["wrong_foreground_class"]
    return errors


def calc_metrics(gt_arrays, pred_arrays, model_name):
    cm = confusion_matrix_from_arrays(gt_arrays, pred_arrays, num_classes=len(CLASS_NAMES))
    metrics = metrics_from_confusion(cm, class_names=CLASS_NAMES)
    rows = flatten_metrics_for_summary(model_name, metrics)

    class_precision = np.array([item["precision"] for item in metrics["per_class"]], dtype=np.float64)
    class_recall = np.array([item["recall"] for item in metrics["per_class"]], dtype=np.float64)
    class_f1 = np.array([item["f1_score"] for item in metrics["per_class"]], dtype=np.float64)
    macro_precision = np.mean(class_precision)
    macro_recall = np.mean(class_recall)
    macro_f1 = np.mean(class_f1)
    supports = np.array([item["support"] for item in metrics["per_class"]], dtype=np.float64)
    weights = supports / supports.sum() if supports.sum() else np.zeros_like(supports)
    rows.append(
        {
            "model": model_name,
            "row_type": "aggregate",
            "class_id": "",
            "class": "macro avg",
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1_score": float(macro_f1),
            "iou": "",
            "dice": "",
            "support": int(supports.sum()),
            "pixel_accuracy": metrics["pixel_accuracy"],
            "global_accuracy": metrics["global_accuracy"],
            "mean_iou": metrics["mean_iou"],
            "mean_dice": metrics["mean_dice"],
        }
    )
    rows.append(
        {
            "model": model_name,
            "row_type": "aggregate",
            "class_id": "",
            "class": "weighted avg",
            "precision": float(np.sum(class_precision * weights)),
            "recall": float(np.sum(class_recall * weights)),
            "f1_score": float(np.sum(class_f1 * weights)),
            "iou": "",
            "dice": "",
            "support": int(supports.sum()),
            "pixel_accuracy": metrics["pixel_accuracy"],
            "global_accuracy": metrics["global_accuracy"],
            "mean_iou": metrics["mean_iou"],
            "mean_dice": metrics["mean_dice"],
        }
    )
    rows.append(
        {
            "model": model_name,
            "row_type": "aggregate",
            "class_id": "",
            "class": "accuracy",
            "precision": metrics["pixel_accuracy"],
            "recall": metrics["pixel_accuracy"],
            "f1_score": metrics["pixel_accuracy"],
            "iou": "",
            "dice": "",
            "support": metrics["support"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "global_accuracy": metrics["global_accuracy"],
            "mean_iou": metrics["mean_iou"],
            "mean_dice": metrics["mean_dice"],
        }
    )

    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return pd.DataFrame(rows), cm, cm_norm, metrics


def per_image_metrics(image_paths, gt_arrays, pred_arrays, model_name):
    rows = []
    for image_path, gt, pred in zip(image_paths, gt_arrays, pred_arrays):
        cm = confusion_matrix_from_arrays([gt], [pred], num_classes=len(CLASS_NAMES))
        metrics = metrics_from_confusion(cm, class_names=CLASS_NAMES)
        row = {
            "model": model_name,
            "image": Path(image_path).name,
            "pixel_accuracy": metrics["pixel_accuracy"],
            "global_accuracy": metrics["global_accuracy"],
            "mean_iou": metrics["mean_iou"],
            "mean_dice": metrics["mean_dice"],
        }
        for item in metrics["per_class"]:
            key = item["class"].lower()
            row[f"iou_{key}"] = item["iou"]
            row[f"dice_{key}"] = item["dice"]
            row[f"precision_{key}"] = item["precision"]
            row[f"recall_{key}"] = item["recall"]
            row[f"f1_{key}"] = item["f1_score"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_confusion_matrix(cm_norm, title, save_path):
    fig, ax = plt.subplots(figsize=(5.0, 4.6), constrained_layout=True)
    image = ax.imshow(cm_norm * 100.0, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            value = cm_norm[i, j] * 100.0
            text = "<0.1" if value < 0.1 else ">99.9" if value > 99.9 else f"{value:.1f}"
            ax.text(j, i, text, ha="center", va="center", color="white" if value > 50 else "black")
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(3), CLASS_LABELS)
    ax.set_yticks(range(3), CLASS_LABELS)
    fig.colorbar(image, ax=ax, label="%")
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def select_representative_crops(gt_arrays, pred_sets, crop_size=400, max_examples=6):
    candidates = []
    step = max(crop_size // 2, 1)
    for image_index, gt in enumerate(gt_arrays):
        h, w = gt.shape
        pred_arrays = [pred_set[image_index] for pred_set in pred_sets]
        for y in range(0, max(h - crop_size + 1, 1), step):
            for x in range(0, max(w - crop_size + 1, 1), step):
                gt_crop = gt[y:y + crop_size, x:x + crop_size]
                plant_score = int((gt_crop > 0).sum())
                mismatch_score = sum(int((pred[y:y + crop_size, x:x + crop_size] != gt_crop).sum()) for pred in pred_arrays)
                score = plant_score + (2 * mismatch_score)
                if score > 0:
                    candidates.append((score, image_index, x, y))

    candidates.sort(reverse=True)
    selected = []
    per_image_count = {}
    for score, image_index, x, y in candidates:
        if per_image_count.get(image_index, 0) >= 2:
            continue
        too_close = any(
            image_index == prev_image and abs(x - prev_x) < crop_size and abs(y - prev_y) < crop_size
            for _, prev_image, prev_x, prev_y in selected
        )
        if too_close:
            continue
        selected.append((score, image_index, x, y))
        per_image_count[image_index] = per_image_count.get(image_index, 0) + 1
        if len(selected) >= max_examples:
            break

    return selected


def add_panel_image(ax, image, title=None):
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")


def plot_difference_legend(ax):
    ax.axis("on")
    grid = np.zeros((3, 3, 3), dtype=np.uint8)
    for true_label in range(3):
        for pred_label in range(3):
            grid[true_label, pred_label] = DIFF_COLORS[(true_label, pred_label)]
    ax.imshow(grid)
    ax.set_xticks(range(3), CLASS_LABELS)
    ax.set_yticks(range(3), CLASS_LABELS)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(length=0)
    ax.set_title("Difference Colors", fontsize=9)


def plot_error_legend(ax):
    ax.axis("off")
    legend_items = [
        ("Correct", ERROR_COLORS["correct"]),
        ("False positive foreground", ERROR_COLORS["false_positive_foreground"]),
        ("False negative foreground", ERROR_COLORS["false_negative_foreground"]),
        ("Wrong foreground class", ERROR_COLORS["wrong_foreground_class"]),
    ]
    for index, (label, color) in enumerate(legend_items):
        y = 1.0 - (index + 1) * 0.22
        ax.add_patch(
            plt.Rectangle(
                (0.05, y),
                0.12,
                0.11,
                transform=ax.transAxes,
                color=np.array(color) / 255.0,
                clip_on=False,
            )
        )
        ax.text(0.22, y + 0.055, label, transform=ax.transAxes, va="center", fontsize=8)
    ax.set_title("Error Map", fontsize=9)


def plot_qualitative_grid(image_arrays, gt_arrays, model_predictions, crop_specs, save_path, crop_size=400):
    if not crop_specs:
        return

    model_names = list(model_predictions.keys())
    rows = ["Image Patch", "Ground Truth"]
    for model_name in model_names:
        rows.append(f"Prediction\n{model_name}")
        rows.append(f"Overlay\n{model_name}")
        rows.append(f"Error Map\n{model_name}")

    n_rows = len(rows)
    n_cols = len(crop_specs)
    fig, axes = plt.subplots(
        n_rows + 1,
        n_cols,
        figsize=(2.2 * n_cols, (2.05 * n_rows) + 1.3),
        squeeze=False,
        gridspec_kw={"height_ratios": [1.0] * n_rows + [0.55]},
    )

    for col, (_, image_index, x, y) in enumerate(crop_specs):
        title = chr(ord("a") + col)
        raw_crop = image_arrays[image_index][y:y + crop_size, x:x + crop_size]
        gt_crop = gt_arrays[image_index][y:y + crop_size, x:x + crop_size]
        add_panel_image(axes[0, col], raw_crop, title=title)
        add_panel_image(axes[1, col], convert_labelmap_to_color(gt_crop, LABEL_COLORS))

        row = 2
        for model_name in model_names:
            pred_crop = model_predictions[model_name][image_index][y:y + crop_size, x:x + crop_size]
            add_panel_image(axes[row, col], convert_labelmap_to_color(pred_crop, LABEL_COLORS))
            add_panel_image(axes[row + 1, col], overlay_prediction(raw_crop, pred_crop))
            add_panel_image(axes[row + 2, col], error_map(gt_crop, pred_crop))
            row += 3

    for row_index, row_label in enumerate(rows):
        axes[row_index, 0].set_ylabel(row_label, rotation=90, fontsize=10)

    for col in range(n_cols):
        axes[-1, col].axis("off")
    plot_error_legend(axes[-1, -1])

    fig.tight_layout(rect=[0.02, 0.02, 1.0, 1.0])
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def write_summary_markdown(save_path, subset, root_path, model_reports, crop_specs):
    lines = [
        f"# Prediction Report: `{subset}`",
        "",
        "This report compares prediction masks against the ground-truth masks in the dataset.",
        "A one-model report is a checkpoint-vs-ground-truth evaluation. A multi-model report uses the same ground truth and compares each listed prediction directory independently.",
        "",
        "## Dataset",
        "",
        f"- Root path: `{root_path}`",
        f"- Subset: `{subset}`",
        f"- Number of evaluated raw captures: {model_reports[0]['num_images'] if model_reports else 0}",
        "",
        "## Models",
        "",
    ]
    for report in model_reports:
        lines.extend(
            [
                f"- `{report['model_name']}`",
                f"  - checkpoint/source: `{report['checkpoint']}`",
                f"  - prediction directory: `{report['prediction_dir']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Label Mapping",
            "",
            "| Label | Class | Visualization color |",
            "| --- | --- | --- |",
            "| BG | Background/soil | gray |",
            "| S | Sorghum | blue |",
            "| W | Weed | orange |",
            "",
            "## Outputs",
            "",
            "- `metrics_summary.csv`: pixel-level precision, recall, F1, IoU, Dice, mIoU, mean Dice, support, and global accuracy.",
            "- `metrics_summary.json`: JSON copy of the pixel-level summary metrics.",
            "- `metrics_per_image.csv`: per-capture pixel accuracy, mIoU, Dice, precision, recall, and F1 scores.",
            "- `confusion_matrix_<model>.csv`: raw confusion matrix counts.",
            "- `confusion_matrix_<model>.png`: normalized confusion matrix.",
            "- `qualitative_grid.png`: raw patch, ground truth, prediction, overlay, and error map visualization.",
            "",
            "`qualitative_grid.png` uses the raw UAV crop as input context, the dataset mask as ground truth, each model mask as prediction, an overlay for visual inspection, and an error map for failure analysis.",
            "Error map colors: black = correct, red = false-positive foreground, blue = false-negative foreground, yellow = wrong foreground class.",
            "",
            "## Representative Crop Coordinates",
            "",
        ]
    )
    for index, (_, image_index, x, y) in enumerate(crop_specs, start=1):
        lines.append(f"- {index}: image_index={image_index}, x={x}, y={y}")

    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_prediction_report(root_path, subset, prediction_entries, report_dir, crop_size=400, max_examples=6):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_per_image = []
    all_metric_json = {}
    model_reports = []
    base_images = None
    base_gts = None
    model_predictions = {}

    for entry in prediction_entries:
        model_name = entry["model_name"]
        prediction_dir = Path(entry["prediction_dir"])
        image_paths, gt_paths, prediction_paths, images, gts, preds = load_prediction_set(root_path, subset, prediction_dir)
        if base_images is None:
            base_images = images
            base_gts = gts
        model_predictions[model_name] = preds

        metrics_df, cm, cm_norm, metrics = calc_metrics(gts, preds, model_name)
        per_image_df = per_image_metrics(image_paths, gts, preds, model_name)
        all_metrics.append(metrics_df)
        all_per_image.append(per_image_df)
        all_metric_json[model_name] = metrics

        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_name)
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(report_dir / f"confusion_matrix_{safe_name}.csv")
        plot_confusion_matrix(cm_norm, model_name, report_dir / f"confusion_matrix_{safe_name}.png")

        model_reports.append(
            {
                "model_name": model_name,
                "checkpoint": entry.get("checkpoint", ""),
                "prediction_dir": str(prediction_dir),
                "num_images": len(image_paths),
                "image_files": [str(path) for path in image_paths],
                "ground_truth_files": [str(path) for path in gt_paths],
                "prediction_files": [str(path) for path in prediction_paths],
                "metrics": metrics,
            }
        )

    metrics_summary_df = pd.concat(all_metrics, ignore_index=True)
    metrics_per_image_df = pd.concat(all_per_image, ignore_index=True)
    metrics_summary_df.to_csv(report_dir / "metrics_summary.csv", index=False)
    metrics_per_image_df.to_csv(report_dir / "metrics_per_image.csv", index=False)
    (report_dir / "metrics_summary.json").write_text(
        json.dumps(all_metric_json, indent=2),
        encoding="utf-8",
    )

    crop_specs = select_representative_crops(
        base_gts,
        [model_predictions[name] for name in model_predictions],
        crop_size=crop_size,
        max_examples=max_examples,
    )
    plot_qualitative_grid(
        base_images,
        base_gts,
        model_predictions,
        crop_specs,
        report_dir / "qualitative_grid.png",
        crop_size=crop_size,
    )

    manifest = {
        "root_path": str(root_path),
        "subset": subset,
        "models": model_reports,
        "report_dir": str(report_dir),
        "crop_size": crop_size,
        "max_examples": max_examples,
        "metrics_summary_json": str(report_dir / "metrics_summary.json"),
    }
    (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary_markdown(report_dir / "README.md", subset, root_path, model_reports, crop_specs)
    return manifest
