import logging
import numpy as np
import torch
from tqdm import tqdm
import kornia
from contextlib import nullcontext
from utils.dataset import UAVDatasetPatches
from torch.utils.data import DataLoader
import random
import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import optuna
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.manual_fcn import load_fcn_resnet
from utils.manual_unet import UNet
from utils.manual_dlplus import DLv3plus
from utils.manual_unet_mobilenetv3 import MobileNetV3UNet
from utils.model_outputs import get_segmentation_logits
from utils.segmentation_metrics import metrics_from_confusion

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def describe_runtime(device):
    print(f"Using device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"CUDA GPU: {gpu_name} ({total_gb:.1f} GiB VRAM)")
        print(f"PyTorch CUDA build: {torch.version.cuda}")


def create_grad_scaler(device, use_amp=True):
    enabled = use_amp and device == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device, use_amp=True):
    enabled = use_amp and device == "cuda"
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda")
    return torch.cuda.amp.autocast()

def get_calculated_means_stds_per_fold(fold):
    means = [
        [0.4895940504368177, 0.4747875829353402, 0.42545172025367883],
        [0.4909516814094245, 0.47507395584447076, 0.4252166750637278],
        [0.4863172918463077, 0.4720067749001233, 0.42307293323046524],
        [0.48556443799258586, 0.471592906257259, 0.42337851381822833]
        ]
    stds = [
        [0.1329905783602554, 0.130645279821384, 0.12234299715980072],
        [0.12910633924968123, 0.12635436744763892, 0.1180632138245313],
        [0.1329739900037901, 0.1304754029316029, 0.12181500603654097],
        [0.1335583288658572, 0.1313047051909438, 0.12297522870807812]
    ]
    return means[fold], stds[fold]

def get_calculated_means_stds_trainval():
    means = [0.48810686542128406, 0.4733653049842984, 0.4242799605915251]
    stds = [0.1321881434144248, 0.12971921686190743, 0.12131885037092494]
    return means, stds

def get_patch_lists(data_path, subset):
    path = data_path / subset / "patches"
    imgPaths = list(path.glob('./img/*.png'))
    img_list = sorted(imgPaths)
    annPaths = list(path.glob('./msk/*.png'))
    msk_list = sorted(annPaths)
    if len(img_list) == 0 or len(msk_list) == 0:
        raise FileNotFoundError(
            f"No patch images/masks found in {path}. Run save_patches.py before training."
        )
    if len(img_list) != len(msk_list):
        raise ValueError(
            f"Patch image/mask count mismatch in {path}: {len(img_list)} images, {len(msk_list)} masks."
        )
    return img_list, msk_list 

def set_study(db_name, study_name, root_path, seed, b_clean_study=False):
    '''
    Creates a new study in a sqlite database located in ./results/
    '''
    sampler = optuna.samplers.TPESampler(seed=seed)
    storage = optuna.storages.RDBStorage(f"sqlite:///{root_path}/results/{db_name}.db", heartbeat_interval=1)
    if b_clean_study:
        print(f"CAUTION: Deleting existing trials in study {study_name}")
        optuna.delete_study(study_name=study_name, storage=f"sqlite:///{root_path}/results/{db_name}.db")
        
    study = optuna.create_study(storage=storage, study_name=study_name, sampler=sampler, direction="minimize", load_if_exists=True)
    return study

def seed_all(seed):
    '''
    sets the initial seed for numpy and pytorch to get reproducible results. 
    One still need to restart the kernel to get reproducible results, as discussed in:
    https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def get_loaders(train_img_dir, train_msk_dir, valid_img_dir ,valid_msk_dir, mean, std, batch_size, num_workers=4, pin_memory=True):
    train_transform = A.Compose(
        [    
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.CLAHE(),
            A.RandomRotate90(),
            A.Transpose(),
            A.Normalize(
                mean = mean,
                std = std,
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    valid_transform = A.Compose(
        [
            A.Normalize(
                mean = mean,
                std = std,
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    train_ds = UAVDatasetPatches(img_list=train_img_dir, msk_list=train_msk_dir, transform=train_transform)
    valid_ds = UAVDatasetPatches(img_list=valid_img_dir, msk_list=valid_msk_dir, transform=valid_transform)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    drop_last_train = len(train_ds) % batch_size == 1
    if drop_last_train:
        print(
            "Dropping the final singleton training batch. "
            "This prevents BatchNorm failures in DeepLabV3+ ASPP pooling when the last batch has one patch."
        )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=drop_last_train, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)

    return train_loader, valid_loader
    
def set_model(architecture, encoder_name, pretrained, b_bilinear, replace_stride_with_dilation, num_classes=3):
    model_name = f"{architecture}_{encoder_name}"
    print(f"MODEL NAME: {model_name}")
    if architecture == "fcn32s":
        if replace_stride_with_dilation:
            model=load_fcn_resnet(encoder_name, 
            num_classes=num_classes, 
            pretrained = pretrained, 
            replace_stride_with_dilation=replace_stride_with_dilation, 
            n_upsample=8, 
            b_bilinear=b_bilinear
            )
        else:
            model=load_fcn_resnet(encoder_name, 
            num_classes=num_classes, 
            pretrained = pretrained, 
            replace_stride_with_dilation=replace_stride_with_dilation, 
            n_upsample=32, 
            b_bilinear=b_bilinear
            )

    elif architecture == "fcn16s":
        model=load_fcn_resnet(encoder_name, 
        num_classes=num_classes, 
        pretrained = pretrained, 
        replace_stride_with_dilation=replace_stride_with_dilation, 
        n_upsample=16, 
        b_bilinear=b_bilinear
        )
    elif architecture == "fcn8s":
        model=load_fcn_resnet(encoder_name, 
        num_classes=num_classes, 
        pretrained = pretrained, 
        replace_stride_with_dilation=replace_stride_with_dilation, 
        n_upsample=8, 
        b_bilinear=b_bilinear
        )
    elif architecture == "unet":
        model = UNet(encoder_name=encoder_name, pretrained=pretrained)
    
    elif architecture == "dlplus":
        model = DLv3plus(encoder_name=encoder_name, pretrained=pretrained, encoder_output_stride=8)
    elif architecture == "unet_mobilenetv3_base":
        model = MobileNetV3UNet(
            encoder_name=encoder_name,
            pretrained=pretrained,
            num_classes=num_classes,
            use_ppm=False,
            use_se=False,
        )
    elif architecture == "unet_mobilenetv3_ppm":
        model = MobileNetV3UNet(
            encoder_name=encoder_name,
            pretrained=pretrained,
            num_classes=num_classes,
            use_ppm=True,
            use_se=False,
        )
    elif architecture == "unet_mobilenetv3":
        model = MobileNetV3UNet(
            encoder_name=encoder_name,
            pretrained=pretrained,
            num_classes=num_classes,
            use_ppm=True,
            use_se=True,
        )
    elif architecture == "unet_mobilenetv3_aux":
        model = MobileNetV3UNet(
            encoder_name=encoder_name,
            pretrained=pretrained,
            num_classes=num_classes,
            use_ppm=True,
            use_se=True,
            use_auxiliary_foreground_head=True,
        )
    else:
        raise NotImplementedError(
            "Specified model is not defined. Implemented architectures are: "
            "fcn8s, fcn16s, fcn32s, unet, dlplus, "
            "unet_mobilenetv3_base, unet_mobilenetv3_ppm, "
            "unet_mobilenetv3, unet_mobilenetv3_aux."
        )
    return model

def save_checkpoint(state, filename="my_ckpt.pth.tar"):
    torch.save(state, filename)
    return

def train_epoch(loader, model, optimizer, loss_fn, scaler, trial_number=None, fold=None, cur_epoch=None, device="cuda", use_amp=True):
    with tqdm(loader, unit="batch", leave=True) as tepoch:
        losses = []
        if fold is not None and trial_number is not None:
            tepoch.set_description(f"Training T{trial_number} F{fold} E{cur_epoch}")
        else:
            tepoch.set_description(f"Retraining E{cur_epoch}")
        for data, targets in tepoch:
            data = data.float().to(device=device, non_blocking=True)
            targets = targets.long().to(device=device, non_blocking=True)
            # forward 
            optimizer.zero_grad()
            with autocast_context(device, use_amp):
                with torch.set_grad_enabled(True):
                    predictions = model(data)
                    loss = loss_fn(predictions, targets)
                # backward
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # update loop
                tepoch.set_postfix(train_loss=loss.item())
                losses.append(loss.item())
            tepoch.set_postfix(train_losses=np.array(losses).mean())
    return float(np.array(losses).mean()) if losses else 0.0


def confusion_matrix_from_logits(predictions, targets, num_classes=3):
    predicted_labels = predictions.argmax(dim=1)
    valid_mask = (targets >= 0) & (targets < num_classes)
    encoded = targets[valid_mask] * num_classes + predicted_labels[valid_mask]
    return torch.bincount(
        encoded.reshape(-1),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes).float()


def f1_loss_from_confusion(confusion, average="macro", class_index=2, class_indices=None, eps=1e-8):
    true_positive = torch.diag(confusion)
    predicted_positive = confusion.sum(dim=0)
    actual_positive = confusion.sum(dim=1)
    precision = true_positive / (predicted_positive + eps)
    recall = true_positive / (actual_positive + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    if average == "macro":
        present_classes = actual_positive > 0
        score = f1[present_classes].mean()
    elif average == "class":
        score = f1[class_index]
    elif average == "selected_macro":
        if class_indices is None:
            raise ValueError("average='selected_macro' requires class_indices.")
        selected_indices = torch.as_tensor(class_indices, device=confusion.device, dtype=torch.long)
        selected_present = actual_positive[selected_indices] > 0
        if not torch.any(selected_present):
            raise ValueError("No selected classes are present in the confusion matrix.")
        score = f1[selected_indices][selected_present].mean()
    else:
        raise ValueError(f"Unsupported F1 average: {average}")

    return 1.0 - score


def validate_epoch(
    loader,
    model,
    cur_epoch,
    fold=None,
    trial_number=None,
    device="cuda",
    use_amp=True,
    loss_fn=None,
    validation_loss="dice",
    return_metrics=False,
):
    valid_loss = 0
    predictions_batches = []
    targets_batches = []
    weighted_losses = []
    sample_count = 0
    f1_confusion = None
    metrics_confusion = None
    model.eval()
    with torch.no_grad():
        with tqdm(loader, unit="batch", leave=False) as tepoch:
            if fold is not None and trial_number is not None:
                tepoch.set_description(f"Validating T{trial_number} F{fold} E{cur_epoch}")
            else:
                tepoch.set_description(f"Validating E{cur_epoch}")
            for idx, (inputs, targets) in enumerate(tepoch):
                inputs = inputs.float().to(device=device, non_blocking=True)
                targets = targets.long().to(device=device, non_blocking=True)
                with autocast_context(device, use_amp):
                    predictions = model(inputs)
                    segmentation_logits = get_segmentation_logits(predictions)
                    batch_confusion = confusion_matrix_from_logits(segmentation_logits, targets)
                    metrics_confusion = (
                        batch_confusion
                        if metrics_confusion is None
                        else metrics_confusion + batch_confusion
                    )
                    if validation_loss == "same":
                        if loss_fn is None:
                            raise ValueError("validation_loss='same' requires loss_fn.")
                        batch_loss = loss_fn(predictions, targets)
                        batch_size = inputs.shape[0]
                        weighted_losses.append(batch_loss.item() * batch_size)
                        sample_count += batch_size
                    elif validation_loss in {"macro_f1", "weed_f1", "foreground_macro_f1"}:
                        f1_confusion = (
                            batch_confusion
                            if f1_confusion is None
                            else f1_confusion + batch_confusion
                        )
                    elif validation_loss == "dice":
                        predictions_batches.append(segmentation_logits.detach().float().cpu())
                        targets_batches.append(targets.detach().cpu())
                    else:
                        raise ValueError(f"Unsupported validation_loss: {validation_loss}")

            if validation_loss == "same":
                valid_loss = np.sum(weighted_losses) / max(sample_count, 1)
            elif validation_loss == "macro_f1":
                valid_loss = f1_loss_from_confusion(f1_confusion, average="macro").item()
            elif validation_loss == "weed_f1":
                valid_loss = f1_loss_from_confusion(
                    f1_confusion,
                    average="class",
                    class_index=2,
                ).item()
            elif validation_loss == "foreground_macro_f1":
                valid_loss = f1_loss_from_confusion(
                    f1_confusion,
                    average="selected_macro",
                    class_indices=(1, 2),
                ).item()
            else:
                predictions_whole = torch.cat(predictions_batches, dim=0)
                targets_whole = torch.cat(targets_batches, dim=0)
                valid_loss = kornia.losses.dice_loss(predictions_whole, targets_whole).item()
    logging.info(f"Validating T{trial_number} F{fold} E{cur_epoch}: valid loss {valid_loss}")
    model.train()
    if return_metrics:
        metric_values = metrics_from_confusion(
            metrics_confusion.detach().cpu().numpy(),
            class_names=["Background", "Sorghum", "Weed"],
        )
        return valid_loss, metric_values
    return valid_loss

