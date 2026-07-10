from multiprocessing import freeze_support
from pathlib import Path

import torch
from skimage import io as skio

from utils.labels import LABEL_COLORS
from utils.parser import ARCHITECTURE_CHOICES, ENCODER_CHOICES, create_test_parser
from utils.patch_utils import get_file_lists, load_image
from utils.predict import get_test_loader, predict, reshape_predictions_to_images
from utils.reporting import generate_prediction_report, model_id_from_path
from utils.train import (
    get_calculated_means_stds_trainval,
    get_patch_lists,
    resolve_device,
    seed_all,
    set_model,
)


def parse_model_config(path):
    name = path.name
    for suffix in (".pth.tar", ".pt", ".pth"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    parts = name.split("_")
    offset = 1 if parts and parts[0] == "model" else 0
    try:
        dil_index = next(i for i in range(offset, len(parts)) if parts[i].startswith("dil"))
        bilin_index = next(i for i in range(dil_index + 1, len(parts)) if parts[i].startswith("bilin"))
    except StopIteration as exc:
        raise ValueError(
            f"Could not parse architecture/configuration from model filename: {path.name}. "
            "Expected model_unet_resnet34_dil0_bilin1_retrained.pt or unet_resnet34_dil0_bilin1_pre1.pth.tar."
        ) from exc

    model_parts = parts[offset:dil_index]
    architecture = None
    encoder_name = None
    for encoder in sorted(ENCODER_CHOICES, key=lambda item: len(item.split("_")), reverse=True):
        encoder_parts = encoder.split("_")
        if len(model_parts) <= len(encoder_parts):
            continue
        if model_parts[-len(encoder_parts):] == encoder_parts:
            candidate_architecture = "_".join(model_parts[:-len(encoder_parts)])
            if candidate_architecture in ARCHITECTURE_CHOICES:
                architecture = candidate_architecture
                encoder_name = encoder
                break

    if architecture is None or encoder_name is None:
        raise ValueError(
            f"Could not parse architecture/encoder from model filename: {path.name}. "
            "Use the training checkpoint naming pattern <architecture>_<encoder>_dil0_bilin1_pre1.pth.tar."
        )

    return {
        "architecture": architecture,
        "encoder_name": encoder_name,
        "replace_stride_with_dilation": parts[dil_index].replace("dil", "") == "1",
        "b_bilinear": parts[bilin_index].replace("bilin", "") == "1",
    }


def resolve_model_path(root_path, model_path_arg):
    model_path = Path(model_path_arg)
    if not model_path.is_absolute():
        model_path = Path(root_path) / model_path

    if model_path.is_file():
        return model_path

    models_dir = Path(root_path) / "models"
    available_models = sorted(
        [
            str(path.relative_to(Path(root_path)))
            for pattern in ("*.pt", "*.pth", "*.pth.tar")
            for path in models_dir.glob(pattern)
        ]
    )
    available_text = (
        "\n".join(f"  - {model}" for model in available_models)
        if available_models
        else "  - no .pt/.pth/.pth.tar files found in models/"
    )
    raise FileNotFoundError(
        f"Model file not found: {model_path}\n"
        "Prediction requires an existing trained checkpoint. Either download/copy the paper model into models/, "
        "or run train.py with --save_checkpoint and pass the generated checkpoint path.\n"
        f"Available local model files:\n{available_text}"
    )


def main():
    args = create_test_parser()

    subset = args.subset
    root_path = args.root_path
    model_save_path = resolve_model_path(root_path, args.model)
    model_name = args.model_name or model_id_from_path(model_save_path)
    batch_size = args.batch_size
    device = resolve_device(args.device)
    use_amp = args.amp
    pin_memory = args.pin_memory if args.pin_memory is not None else device == "cuda"

    seed_all(seed=args.seed)
    print(f"Using Seed {args.seed}")

    data_path = Path(root_path) / "data"
    model_config = parse_model_config(model_save_path)
    architecture = model_config["architecture"]
    encoder_name = model_config["encoder_name"]
    replace_stride_with_dilation = model_config["replace_stride_with_dilation"]
    b_bilinear = model_config["b_bilinear"]

    path_to_save = Path(args.output_dir) if args.output_dir else Path(root_path) / "results" / "predictions" / subset
    path_to_save.mkdir(parents=True, exist_ok=True)

    test_imgs, test_msks = get_patch_lists(
        data_path=data_path,
        subset=subset,
    )

    test_complete_img_ls, _ = get_file_lists(
        data_path,
        subset=subset,
    )

    img_shape = load_image(path=str(test_complete_img_ls[0])).shape
    means, stds = get_calculated_means_stds_trainval()

    test_loader = get_test_loader(
        test_img_dir=test_imgs,
        test_msk_dir=test_msks,
        mean=means,
        std=stds,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    loaded_model = torch.load(model_save_path, map_location=device)
    print(f"Loading: {architecture} {encoder_name} ...")
    print(f"Report model name: {model_name}")
    model = set_model(
        architecture=architecture,
        encoder_name=encoder_name,
        pretrained=False,
        b_bilinear=b_bilinear,
        replace_stride_with_dilation=replace_stride_with_dilation,
        num_classes=3,
    ).to(device=device)
    model.load_state_dict(loaded_model["state_dict"])
    print(f"Loaded checkpoint: {model_save_path}")

    print("Predicting...")
    preds = predict(
        model=model,
        test_loader=test_loader,
        device=device,
        use_amp=use_amp,
    )

    print("Combining Slices...")
    colored_predictions = reshape_predictions_to_images(
        preds=preds,
        labels=LABEL_COLORS,
        mask_shape=img_shape[:2],
    )
    print(f"Saving Predictions to {path_to_save}...")
    for preds_to_save, img_name in zip(colored_predictions, test_complete_img_ls):
        skio.imsave(path_to_save / f"{img_name.stem}_pred.png", preds_to_save, check_contrast=False)

    if args.report:
        report_dir = (
            Path(args.report_dir)
            if args.report_dir
            else Path(root_path) / "results" / "reports" / subset / model_name
        )
        print(f"Generating prediction report in {report_dir}...")
        generate_prediction_report(
            root_path=root_path,
            subset=subset,
            prediction_entries=[
                {
                    "model_name": model_name,
                    "prediction_dir": str(path_to_save),
                    "checkpoint": str(model_save_path),
                }
            ],
            report_dir=report_dir,
            crop_size=args.crop_size,
            max_examples=args.max_examples,
        )
        print(f"Report saved to {report_dir}")


if __name__ == "__main__":
    freeze_support()
    main()
