import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import math 
import shutil
import time
from utils.losses import create_loss, estimate_class_weights, foreground_weights_from_class_counts
from utils.training_logs import write_training_log_outputs

from utils.train import (
    seed_all,
    set_study,
    set_model,
    resolve_device,
    describe_runtime,
    create_grad_scaler,
    get_calculated_means_stds_per_fold, 
    get_patch_lists, 
    get_loaders, 
    save_checkpoint,
    train_epoch,
    validate_epoch,
)
from utils.parser import create_train_parser


def objective(trial):
    epochs_no_improve:int = 0
    kfold = KFold(n_splits=num_folds, shuffle=False)
    loss_total = np.ones(num_folds)*99999
    epochs = np.ones(num_folds)*0
    img_list, msk_list = get_patch_lists(
    data_path=data_path, 
    subset="trainval")
    for fold, (train_ids, val_ids) in enumerate(kfold.split(img_list)):
        train_img_dir = [img_list[i] for i in train_ids]
        train_msk_dir = [msk_list[i] for i in train_ids]
        valid_img_dir = [img_list[i] for i in val_ids]
        valid_msk_dir = [msk_list[i] for i in val_ids]
        epochs_no_improve = 0

        model = set_model(architecture=architecture, encoder_name=encoder_name, pretrained=pretrained, b_bilinear=b_bilinear, replace_stride_with_dilation=replace_stride_with_dilation, num_classes=3).to(device=device)
        
        class_weights = None
        class_counts = None
        foreground_class_weights = None
        if class_weights_mode == "auto":
            class_weights, class_counts = estimate_class_weights(
                train_msk_dir,
                num_classes=3,
                max_weight=class_weight_max,
                strategy=class_weight_strategy,
            )
            print(f"Class counts for fold {fold}: {class_counts.astype(int).tolist()}")
            print(f"Class weight strategy for fold {fold}: {class_weight_strategy}")
            print(f"Class weights for fold {fold}: {class_weights.tolist()}")
            class_weights = class_weights.to(device=device)
            foreground_class_weights = foreground_weights_from_class_counts(
                class_counts,
                max_weight=class_weight_max,
                strategy=class_weight_strategy,
            ).to(device=device)
            print(f"Foreground auxiliary weights for fold {fold}: {foreground_class_weights.tolist()}")

        loss_fn = create_loss(
            loss_name=loss_name,
            class_weights=class_weights,
            foreground_class_weights=foreground_class_weights,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            foreground_aux_weight=foreground_aux_weight,
        )
        lr = trial.suggest_float("lr", lr_ranges[0], lr_ranges[1], log=True)
        print(f"suggested LR: {lr}")
        reduce_factor = trial.suggest_int("lr_factor", int(lr_factor_ranges[0]*10), int(lr_factor_ranges[1]*10), step=int(lr_factor_ranges[2]*10))
        reduce_factor = reduce_factor*0.1
        optimizer = optim.Adam(model.parameters(), lr = lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=reduce_factor, min_lr=lr_ranges[0], patience=lr_scheduler_patience)
        means, stds = get_calculated_means_stds_per_fold(fold)
        train_loader, valid_loader = get_loaders(
            train_img_dir = train_img_dir,
            train_msk_dir = train_msk_dir,
            valid_img_dir = valid_img_dir, 
            valid_msk_dir = valid_msk_dir,
            mean = means,
            std = stds,
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = pin_memory,
        )
        scaler = create_grad_scaler(device=device, use_amp=use_amp)
        fold_final_epoch = -1
        for epoch in range(max_epochs):
            fold_final_epoch = epoch
            epoch_start = time.perf_counter()
            epoch_lr = optimizer.param_groups[0]["lr"]
            train_loss = train_epoch(
                train_loader, 
                model, 
                optimizer, 
                loss_fn, 
                scaler, 
                cur_epoch=epoch,
                trial_number=trial.number,
                fold=fold,
                device=device,
                use_amp=use_amp,
                )
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            
            valid_loss, validation_metrics = validate_epoch(
                valid_loader, 
                model, 
                cur_epoch=epoch, 
                trial_number=trial.number,
                fold=fold,
                device=device,
                use_amp=use_amp,
                loss_fn=loss_fn,
                validation_loss=validation_loss_mode,
                return_metrics=True,
                )
            # Keep scheduler/checkpoint logic numeric if a validation batch produces NaN.
            if math.isnan(valid_loss):
                print("Validation loss is NaN; using 99999 for scheduler and study logging.")
                valid_loss = 99999
            scheduler.step(valid_loss)
            epoch_time_sec = time.perf_counter() - epoch_start

            training_log_rows.append(
                {
                    "run_prefix": run_prefix,
                    "db_name": db_name,
                    "study_name": study_name,
                    "architecture": architecture,
                    "encoder_name": encoder_name,
                    "trial": trial.number,
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "valid_loss": float(valid_loss),
                    "validation_mean_iou": validation_metrics["mean_iou"],
                    "validation_mean_dice": validation_metrics["mean_dice"],
                    "validation_pixel_accuracy": validation_metrics["pixel_accuracy"],
                    "learning_rate": float(epoch_lr),
                    "epoch_time_sec": float(epoch_time_sec),
                    "loss": loss_name,
                    "class_weights": class_weights_mode,
                    "class_weight_strategy": class_weight_strategy,
                    "ce_weight": ce_weight,
                    "dice_weight": dice_weight,
                    "foreground_aux_weight": foreground_aux_weight,
                    "validation_loss_mode": validation_loss_mode,
                    "batch_size": batch_size,
                    "device": device,
                    "use_amp": use_amp,
                }
            )
            
            if valid_loss < loss_total[fold]:
                loss_total[fold] = valid_loss
                epochs_no_improve = 0
                if b_save_checkpoint:
                    checkpoint_path = get_trial_checkpoint_path(trial.number)
                    save_checkpoint(checkpoint, filename=checkpoint_path)
                    trial.set_user_attr('checkpoint_path', str(checkpoint_path))
            else:
                epochs_no_improve+=1
            
            if epochs_no_improve >= es_patience:
                print(f"Early Stopping on epoch {epoch}")
                break
        epochs[fold] = fold_final_epoch
        if device == "cuda":
            torch.cuda.empty_cache()

    trial.set_user_attr('Valid loss per fold', list(loss_total))
    trial.set_user_attr('root path', root_path)
    trial.set_user_attr('architecture', architecture)
    trial.set_user_attr('encoder_name', encoder_name)
    trial.set_user_attr('batch_size', batch_size)
    trial.set_user_attr('b_bilinear', b_bilinear)
    trial.set_user_attr('pretrained', pretrained)
    trial.set_user_attr('replace_stride', replace_stride_with_dilation)
    trial.set_user_attr('final_epoch', list(epochs))
    trial.set_user_attr('lr_scheduler_patience', lr_scheduler_patience)
    trial.set_user_attr('loss', loss_name)
    trial.set_user_attr('class_weights', class_weights_mode)
    trial.set_user_attr('class_weight_strategy', class_weight_strategy)
    trial.set_user_attr('ce_weight', ce_weight)
    trial.set_user_attr('dice_weight', dice_weight)
    trial.set_user_attr('foreground_aux_weight', foreground_aux_weight)
    trial.set_user_attr('validation_loss', validation_loss_mode)
    print(f"Validation loss per fold: {loss_total}")  
    return np.mean(loss_total)


def get_trial_checkpoint_path(trial_number):
    if n_trials <= 1:
        return canonical_checkpoint_path
    return trial_checkpoint_path / f"trial_{trial_number:03d}_{checkpoint_filename}"


def publish_best_trial_checkpoint(study):
    if not b_save_checkpoint or n_trials <= 1:
        return

    ranked_trials = sorted(
        [
            trial
            for trial in study.trials
            if trial.value is not None and trial.user_attrs.get('checkpoint_path')
        ],
        key=lambda trial: trial.value,
    )
    if not ranked_trials:
        print("No trial checkpoint metadata was found; canonical checkpoint was not updated.")
        return

    best_trial = ranked_trials[0]
    best_checkpoint_path = Path(best_trial.user_attrs['checkpoint_path'])
    if not best_checkpoint_path.is_file():
        print(f"Best trial checkpoint was not found: {best_checkpoint_path}")
        return

    shutil.copy2(best_checkpoint_path, canonical_checkpoint_path)
    print(
        f"Published best available trial checkpoint: {canonical_checkpoint_path} "
        f"(trial={best_trial.number}, value={best_trial.value})"
    )


if __name__ == "__main__":
    args = create_train_parser()
    run_prefix:str = args.run_prefix
    b_clean_study:bool = args.b_clean_study
    b_save_checkpoint:bool = args.save_checkpoint
    pretrained:bool = args.pretrained
    b_bilinear:bool = args.b_bilinear
    replace_stride_with_dilation:bool = args.replace_stride_with_dilation
    encoder_name:str = args.encoder_name
    architecture:str = args.architecture
    lr_ranges = [args.lr_min, args.lr_max]

    if args.db_name == "":
        db_name:str = f"{run_prefix}_{architecture}_{encoder_name}_dil{int(replace_stride_with_dilation)}_bilin{int(b_bilinear)}_pre{int(pretrained)}"
    else:
        db_name = args.db_name
    if args.study_name == "":
        study_name:str = f"{architecture}_{encoder_name}_dil{int(replace_stride_with_dilation)}_bilin{int(b_bilinear)}_pre{int(pretrained)}"
    else:
        study_name = args.study_name
    root_path: str = args.root_path
    data_path = Path(root_path) / "data" 
    num_folds:int = args.n_folds
    batch_size:int = args.batch_size
    n_trials:int = args.n_trials

    lr_factor_ranges = [0.1, 0.9, 0.1]
    max_epochs:int = args.max_epochs
    es_patience:int = args.early_stop_patience
    lr_scheduler_patience:int = args.lr_scheduler_patience
    loss_name: str = args.loss
    class_weights_mode: str = args.class_weights
    class_weight_strategy: str = args.class_weight_strategy
    class_weight_max: float = args.class_weight_max
    ce_weight: float = args.ce_weight
    dice_weight: float = args.dice_weight
    foreground_aux_weight: float = args.foreground_aux_weight
    validation_loss_mode: str = args.validation_loss
    seed:int = args.seed

    device: str = resolve_device(args.device)
    use_amp: bool = args.amp
    num_workers: int = args.num_workers
    pin_memory: bool = args.pin_memory if args.pin_memory is not None else device == "cuda"

    seed_all(seed=seed)
    describe_runtime(device)

    # Create Paths
    model_path = Path(root_path) / "models"
    model_path.mkdir(parents=True, exist_ok=True)
    result_path = Path(root_path) / "results"
    result_path.mkdir(parents=True, exist_ok=True)
    checkpoint_filename = (
        f"{architecture}_{encoder_name}_dil{int(replace_stride_with_dilation)}_"
        f"bilin{int(b_bilinear)}_pre{int(pretrained)}.pth.tar"
    )
    canonical_checkpoint_path = model_path / checkpoint_filename
    trial_checkpoint_path = model_path / "_trial_checkpoints" / (
        f"{run_prefix}_{architecture}_{encoder_name}_dil{int(replace_stride_with_dilation)}_"
        f"bilin{int(b_bilinear)}_pre{int(pretrained)}"
    )
    if b_save_checkpoint and n_trials > 1:
        trial_checkpoint_path.mkdir(parents=True, exist_ok=True)

    study = set_study(db_name=db_name, study_name=study_name, root_path=root_path, seed=seed, b_clean_study=b_clean_study)
    training_log_rows = []

    study.optimize(lambda trial: objective(trial), n_trials=n_trials)
    publish_best_trial_checkpoint(study)
    if training_log_rows:
        log_outputs = write_training_log_outputs(
            training_log_rows,
            output_dir=Path(root_path) / "results" / "training_logs",
            run_id=f"{db_name}_{study_name}",
        )
        print(f"Training log CSV: {log_outputs['csv']}")
        print(f"Training log JSON: {log_outputs['json']}")
