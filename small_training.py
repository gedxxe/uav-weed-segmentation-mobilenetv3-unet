import torch
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import KFold
from utils.losses import create_loss, estimate_class_weights

from utils.train import (
    seed_all,
    set_model,
    resolve_device,
    describe_runtime,
    create_grad_scaler,
    get_calculated_means_stds_per_fold,
    get_patch_lists,
    get_loaders,
    train_epoch,
    validate_epoch,
)
from utils.parser import create_train_parser


def main():
    args = create_train_parser()
    seed_all(seed=args.seed)

    device = resolve_device(args.device)
    use_amp = args.amp
    pin_memory = args.pin_memory if args.pin_memory is not None else device == "cuda"
    describe_runtime(device)

    data_path = Path(args.root_path) / "data"
    img_list, msk_list = get_patch_lists(data_path=data_path, subset="trainval")
    kfold = KFold(n_splits=args.n_folds, shuffle=False)
    train_ids, val_ids = next(kfold.split(img_list))

    train_img_dir = [img_list[i] for i in train_ids]
    train_msk_dir = [msk_list[i] for i in train_ids]
    valid_img_dir = [img_list[i] for i in val_ids]
    valid_msk_dir = [msk_list[i] for i in val_ids]

    model = set_model(
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        pretrained=args.pretrained,
        b_bilinear=args.b_bilinear,
        replace_stride_with_dilation=args.replace_stride_with_dilation,
        num_classes=3,
    ).to(device=device)

    class_weights = None
    if args.class_weights == "auto":
        class_weights, class_counts = estimate_class_weights(
            train_msk_dir,
            num_classes=3,
            max_weight=args.class_weight_max,
        )
        print(f"Class counts for smoke fold: {class_counts.astype(int).tolist()}")
        print(f"Class weights for smoke fold: {class_weights.tolist()}")
        class_weights = class_weights.to(device=device)

    loss_fn = create_loss(
        loss_name=args.loss,
        class_weights=class_weights,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr_min)
    means, stds = get_calculated_means_stds_per_fold(0)
    train_loader, valid_loader = get_loaders(
        train_img_dir=train_img_dir,
        train_msk_dir=train_msk_dir,
        valid_img_dir=valid_img_dir,
        valid_msk_dir=valid_msk_dir,
        mean=means,
        std=stds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    scaler = create_grad_scaler(device=device, use_amp=use_amp)
    for epoch in range(args.max_epochs):
        train_loss = train_epoch(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler,
            cur_epoch=epoch,
            trial_number=0,
            fold=0,
            device=device,
            use_amp=use_amp,
        )
        valid_loss = validate_epoch(
            valid_loader,
            model,
            cur_epoch=epoch,
            trial_number=0,
            fold=0,
            device=device,
            use_amp=use_amp,
            loss_fn=loss_fn,
            validation_loss=args.validation_loss,
        )
        print(f"{train_loss=}, {valid_loss=}")

    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
