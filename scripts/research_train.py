"""Leakage-aware experiment pipeline: grouped CV (Optuna) and final fit.

Run from repository root. Original train.py/retrain.py and checkpoints are untouched.
Run 'cv' first. Run 'final' only after CV is complete. The test set is never opened here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import optuna
import torch
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.losses import create_loss, estimate_class_weights, foreground_weights_from_class_counts
from utils.train import (
    create_grad_scaler, get_loaders, resolve_device, seed_all, set_model,
    train_epoch, validate_epoch,
)


def get_manifest(root):
    manifest = root / "data" / "trainval" / "research_patches" / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing {manifest}; run scripts/research_data.py first")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty patch manifest")
    seen = set()
    for row in rows:
        if row["patch_image"] in seen:
            raise ValueError("Duplicate patch path in manifest")
        seen.add(row["patch_image"])
        for name in ("patch_image", "patch_mask"):
            if not (root / row[name]).is_file():
                raise FileNotFoundError(root / row[name])
    return rows


def split_grouped(rows, n_folds):
    groups = np.array([r["source_image"] for r in rows])
    if len(set(groups)) < n_folds:
        raise ValueError(f"Need >= {n_folds} independent sources, got {len(set(groups))}")
    for fold, (train_idx, val_idx) in enumerate(
        GroupKFold(n_splits=n_folds).split(np.arange(len(rows)), groups=groups)
    ):
        train_sources = set(groups[train_idx])
        val_sources = set(groups[val_idx])
        if train_sources & val_sources:
            raise AssertionError("Data leakage: source occurs in both subsets")
        train = [rows[i] for i in train_idx if int(rows[i]["train_eligible"]) == 1]
        val = [rows[i] for i in val_idx]  # no validation filtering
        if not train or not val:
            raise ValueError(f"Fold {fold}: empty training or validation subset")
        yield fold, train, val, train_sources, val_sources


def normalize_stats(root, rows):
    """Calculate per-channel mean/std from TRAINING pixels only, excluding padding."""
    sums = np.zeros(3, dtype=np.float64)
    sums2 = np.zeros(3, dtype=np.float64)
    count = 0
    for row in rows:
        img = cv2.imread(str(root / row["patch_image"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["patch_image"])
        h, w = int(row["valid_height"]), int(row["valid_width"])
        rgb = cv2.cvtColor(img[:h, :w], cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        flattened = rgb.reshape(-1, 3)
        sums += flattened.sum(axis=0)
        sums2 += (flattened * flattened).sum(axis=0)
        count += flattened.shape[0]
    if not count:
        raise ValueError("No training pixels for normalization")
    mean = sums / count
    std = np.sqrt(np.maximum(sums2 / count - mean * mean, 1e-12))
    return mean.tolist(), std.tolist()


def paths(root, rows):
    return (
        [root / r["patch_image"] for r in rows],
        [root / r["patch_mask"] for r in rows],
    )


def loss_for_rows(root, rows, conf, device):
    weights = None
    foreground = None
    if conf["class_weights"] == "auto":
        _, masks = paths(root, rows)
        weights, counts = estimate_class_weights(
            masks, num_classes=3, max_weight=conf["class_weight_max"],
            strategy=conf["class_weight_strategy"],
        )
        foreground = foreground_weights_from_class_counts(
            counts, max_weight=conf["class_weight_max"],
            strategy=conf["class_weight_strategy"],
        ).to(device)
        weights = weights.to(device)
    return create_loss(
        conf["loss"], class_weights=weights, foreground_class_weights=foreground,
        foreground_aux_weight=conf["aux_weight"],
    )


def make_model(conf, device):
    return set_model(
        architecture=conf["architecture"], encoder_name=conf["encoder"],
        pretrained=True, b_bilinear=True,
        replace_stride_with_dilation=False, num_classes=3,
    ).to(device)


def write_csv(filename, rows):
    if not rows:
        return
    with Path(filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cv_command(args):
    root = args.root.resolve()
    rows = get_manifest(root)
    fold_data = list(split_grouped(rows, args.folds))
    out = root / "results" / "research_runs" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    conf_path = out / "cv_config.json"
    conf = {
        "architecture": args.architecture,
        "encoder": args.encoder,
        "loss": args.loss,
        "class_weights": args.class_weights,
        "class_weight_strategy": args.class_weight_strategy,
        "class_weight_max": args.class_weight_max,
        "aux_weight": args.aux_weight,
        "validation_loss": args.validation_loss,
        "folds": args.folds,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "manifest": "data/trainval/research_patches/manifest.csv",
    }
    if conf_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{out} already contains a CV run. Choose another --run-id or use --resume."
            )
        previous = json.loads(conf_path.read_text(encoding="utf-8"))
        if previous != conf:
            raise ValueError("Resume configuration differs from the original CV run")
    else:
        conf_path.write_text(json.dumps(conf, indent=2), encoding="utf-8")

    split_rows = []
    fold_statistics = []
    for fold, train, val, train_sources, val_sources in fold_data:
        for part, sample in (("train", train), ("validation", val)):
            split_rows.extend({
                "fold": fold, "subset": part, "source_image": r["source_image"],
                "patch_image": r["patch_image"],
            } for r in sample)
        mean, std = normalize_stats(root, train)
        fold_statistics.append((mean, std))
        print(
            f"Fold {fold}: train {len(train)} patches / {sorted(train_sources)}, "
            f"valid {len(val)} patches / {sorted(val_sources)}, mean={mean}, std={std}"
        )
    write_csv(out / "splits.csv", split_rows)
    device = resolve_device(args.device)
    use_amp = (device == "cuda" and not args.no_amp)
    db_path = (out / "optuna.db").resolve()
    db_url = f"sqlite:///{db_path.as_posix()}"
    study = optuna.create_study(
        study_name=args.run_id, direction="minimize",
        storage=db_url, load_if_exists=args.resume,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    log_rows = []

    def objective(trial):
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        fold_results = []
        for fold, train, val, _, _ in fold_data:
            seed_all(args.seed + fold)
            model = make_model(conf, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )
            loss_fn = loss_for_rows(root, train, conf, device)
            mean, std = fold_statistics[fold]
            train_img, train_mask = paths(root, train)
            val_img, val_mask = paths(root, val)
            train_loader, val_loader = get_loaders(
                train_img, train_mask, val_img, val_mask,
                mean, std, args.batch_size, args.workers, device == "cuda",
            )
            scaler = create_grad_scaler(device=device, use_amp=use_amp)
            best_loss = math.inf
            best_epoch = 0
            best_metrics = {}
            epochs_without_improvement = 0
            ckpt_dir = out / "cv_checkpoints" / f"trial_{trial.number:03d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = ckpt_dir / f"fold_{fold}.pt"
            for epoch in range(args.epochs):
                train_loss = train_epoch(
                    train_loader, model, optimizer, loss_fn, scaler,
                    trial_number=trial.number, fold=fold, cur_epoch=epoch,
                    device=device, use_amp=use_amp,
                )
                val_loss, metrics = validate_epoch(
                    val_loader, model, cur_epoch=epoch,
                    trial_number=trial.number, fold=fold,
                    device=device, use_amp=use_amp,
                    loss_fn=loss_fn,
                    validation_loss=args.validation_loss,
                    return_metrics=True,
                )
                scheduler.step()
                if not np.isfinite(val_loss):
                    raise ValueError("Non-finite validation loss")
                log_rows.append({
                    "trial": trial.number, "fold": fold, "epoch": epoch + 1,
                    "train_loss": train_loss, "validation_loss": val_loss,
                    "validation_miou": metrics["mean_iou"],
                    "validation_dice": metrics["mean_dice"],
                })
                write_csv(out / "epoch_log.csv", log_rows)
                if val_loss < best_loss - 1e-6:
                    best_loss = float(val_loss)
                    best_epoch = epoch + 1
                    best_metrics = metrics
                    epochs_without_improvement = 0
                    torch.save({
                        "state_dict": model.state_dict(),
                        "mean": mean, "std": std,
                        "fold": fold, "trial": trial.number,
                        "epoch": best_epoch, "lr": lr,
                        "architecture": conf["architecture"], "encoder": conf["encoder"],
                    }, checkpoint_path)
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    break
            fold_results.append({
                "fold": fold, "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "best_validation_miou": best_metrics.get("mean_iou"),
                "checkpoint": str(checkpoint_path.relative_to(root).as_posix()),
            })
            del model, optimizer, scheduler, scaler, train_loader, val_loader
            if device == "cuda":
                torch.cuda.empty_cache()
        trial.set_user_attr("fold_results", fold_results)
        return float(np.mean([r["best_validation_loss"] for r in fold_results]))

    remaining = max(0, args.trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining)
    if not study.best_trials:
        raise RuntimeError("No successful Optuna trial")
    best = study.best_trial
    summary = {
        "run_id": args.run_id,
        "best_trial": best.number,
        "best_params": best.params,
        "best_cv_loss": best.value,
        "fold_results": best.user_attrs["fold_results"],
        "final_epochs": int(round(statistics.median(
            [r["best_epoch"] for r in best.user_attrs["fold_results"]]
        ))),
        "source_groups": sorted({r["source_image"] for r in rows}),
        "note": "No test images or test labels used for CV or tuning.",
    }
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def final_command(args):
    root = args.root.resolve()
    out = root / "results" / "research_runs" / args.run_id
    conf = json.loads((out / "cv_config.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "cv_summary.json").read_text(encoding="utf-8"))
    rows = [r for r in get_manifest(root) if int(r["train_eligible"]) == 1]
    if not rows:
        raise ValueError("No eligible training patches")
    device = resolve_device(args.device)
    use_amp = (device == "cuda" and not args.no_amp)
    seed_all(conf["seed"])
    model = make_model(conf, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=summary["best_params"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=conf["max_epochs"]
    )
    loss_fn = loss_for_rows(root, rows, conf, device)
    mean, std = normalize_stats(root, rows)
    train_img, train_mask = paths(root, rows)
    # This method only constructs the training loader; validation is NOT run.
    train_loader, _ = get_loaders(
        train_img, train_mask, train_img, train_mask,
        mean, std, conf["batch_size"], args.workers, device == "cuda",
    )
    scaler = create_grad_scaler(device=device, use_amp=use_amp)
    epochs = summary["final_epochs"]
    if not (1 <= epochs <= conf["max_epochs"]):
        raise ValueError("Unexpected final training epoch count")
    final_logs = []
    for epoch in range(epochs):
        loss = train_epoch(
            train_loader, model, optimizer, loss_fn, scaler,
            cur_epoch=epoch, device=device, use_amp=use_amp,
        )
        scheduler.step()
        final_logs.append({"epoch": epoch + 1, "train_loss": loss})
        write_csv(out / "final_epoch_log.csv", final_logs)
    model_dir = root / "models" / "research_runs" / args.run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"{conf['architecture']}_{conf['encoder']}_dil0_bilin1_pre1.pth.tar"
    checkpoint_path = model_dir / model_name
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(f"{checkpoint_path} exists; --overwrite to replace")
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean, "std": std, "training_sources": sorted({r["source_image"] for r in rows}),
        "architecture": conf["architecture"], "encoder": conf["encoder"],
        "run_id": args.run_id, "epochs": epochs,
        "loss": conf["loss"], "best_lr": summary["best_params"]["lr"],
    }, checkpoint_path)
    record = {
        "checkpoint": str(checkpoint_path.relative_to(root).as_posix()),
        "epochs": epochs, "patches": len(rows), "mean": mean, "std": std,
        "note": "Final fit uses training+validation captures only; test remains untouched.",
    }
    (out / "final_summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cv = sub.add_parser("cv", help="4-fold GroupKFold + Optuna; never reads test set")
    cv.add_argument("--architecture", required=True)
    cv.add_argument("--encoder", required=True)
    cv.add_argument("--loss", choices=("dice", "ce", "ce_dice", "ce_dice_aux_foreground"), default="dice")
    cv.add_argument("--class-weights", choices=("none", "auto"), default="none")
    cv.add_argument("--class-weight-strategy", choices=("inverse_frequency", "sqrt_inverse"), default="inverse_frequency")
    cv.add_argument("--class-weight-max", type=float, default=8.0)
    cv.add_argument("--aux-weight", type=float, default=0.3)
    cv.add_argument("--validation-loss", choices=("dice", "macro_f1", "weed_f1", "foreground_macro_f1"), default="dice")
    cv.add_argument("--folds", type=int, default=4)
    cv.add_argument("--trials", type=int, default=10, help="Total target number of trials, including resumed trials")
    cv.add_argument("--epochs", type=int, default=150)
    cv.add_argument("--patience", type=int, default=10)
    cv.add_argument("--batch-size", type=int, default=4)
    cv.add_argument("--lr-min", type=float, default=1e-4)
    cv.add_argument("--lr-max", type=float, default=1e-2)
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--resume", action="store_true")
    final = sub.add_parser("final", help="Train on all trainval using the selected CV recipe")
    final.add_argument("--overwrite", action="store_true")
    for command in (cv, final):
        command.add_argument("--root", type=Path, default=ROOT)
        command.add_argument("--run-id", required=True)
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        command.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.command == "cv":
        if args.trials < 1 or args.epochs < 1 or args.patience < 1 or args.folds < 2 or args.lr_min <= 0 or args.lr_max < args.lr_min:
            parser.error("Invalid CV configuration")
        if args.loss == "ce_dice_aux_foreground" and args.architecture != "unet_mobilenetv3_aux":
            parser.error("Auxiliary loss requires unet_mobilenetv3_aux")
        if args.architecture == "unet_mobilenetv3_aux" and args.loss != "ce_dice_aux_foreground":
            parser.error("For a valid auxiliary ablation use ce_dice_aux_foreground")
        cv_command(args)
    else:
        final_command(args)


if __name__ == "__main__":
    main()
