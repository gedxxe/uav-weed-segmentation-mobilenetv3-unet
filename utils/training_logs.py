import json
import os
import re
from pathlib import Path

_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd


def safe_run_id(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "training_run"


def write_training_log_outputs(rows, output_dir, run_id):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = safe_run_id(run_id)

    csv_path = output_dir / f"{run_id}_training_log.csv"
    json_path = output_dir / f"{run_id}_training_log.json"
    summary_path = output_dir / f"{run_id}_training_summary.json"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    total_time = float(df["epoch_time_sec"].sum()) if "epoch_time_sec" in df and not df.empty else 0.0
    summary = {
        "run_id": run_id,
        "epochs_logged": int(len(df)),
        "total_training_time_sec": total_time,
        "total_training_time_minutes": total_time / 60.0,
        "csv": str(csv_path),
        "json": str(json_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_paths = plot_training_curves(df, output_dir, run_id)
    return {
        "csv": csv_path,
        "json": json_path,
        "summary": summary_path,
        "plots": plot_paths,
    }


def plot_training_curves(df, output_dir, run_id):
    output_dir = Path(output_dir)
    plot_specs = [
        ("train_loss", "Train Loss", "train_loss_curve.png"),
        ("valid_loss", "Validation Loss", "validation_loss_curve.png"),
        ("validation_mean_iou", "Validation mIoU", "validation_miou_curve.png"),
        ("validation_mean_dice", "Validation Mean Dice", "validation_dice_curve.png"),
        ("learning_rate", "Learning Rate", "learning_rate_curve.png"),
    ]

    paths = {}
    if df.empty:
        return paths

    for column, title, filename in plot_specs:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        for (trial, fold), group in df.assign(_metric=numeric).groupby(["trial", "fold"]):
            group = group.sort_values("epoch")
            ax.plot(
                group["epoch"],
                group["_metric"],
                marker="o",
                linewidth=1.2,
                markersize=2.5,
                label=f"T{trial} F{fold}",
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=7, ncol=2)

        save_path = output_dir / f"{run_id}_{filename}"
        fig.savefig(save_path, dpi=180)
        plt.close(fig)
        paths[column] = save_path

    return paths
