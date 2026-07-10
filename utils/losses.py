import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn

from utils.model_outputs import get_foreground_logits, get_segmentation_logits


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, class_weights=None, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = kornia.losses.DiceLoss()

    def forward(self, predictions, targets):
        predictions = get_segmentation_logits(predictions)
        return (
            self.ce_weight * self.cross_entropy(predictions, targets)
            + self.dice_weight * self.dice(predictions, targets)
        )


class AuxiliaryForegroundLoss(nn.Module):
    def __init__(
        self,
        primary_loss,
        foreground_aux_weight=0.3,
        foreground_class_weights=None,
    ):
        super().__init__()
        if foreground_aux_weight < 0:
            raise ValueError("foreground_aux_weight must be non-negative.")
        self.primary_loss = primary_loss
        self.foreground_aux_weight = foreground_aux_weight
        self.foreground_cross_entropy = nn.CrossEntropyLoss(weight=foreground_class_weights)

    def forward(self, predictions, targets):
        primary = self.primary_loss(predictions, targets)
        if self.foreground_aux_weight == 0:
            return primary

        foreground_logits = get_foreground_logits(predictions)
        foreground_targets = (targets > 0).long()
        foreground = self.foreground_cross_entropy(foreground_logits, foreground_targets)
        return primary + self.foreground_aux_weight * foreground


def weights_from_counts(counts, max_weight=5.0, strategy="inverse_frequency"):
    if max_weight <= 0:
        raise ValueError("max_weight must be greater than 0.")
    counts = np.asarray(counts, dtype=np.float64)
    if np.any(counts == 0):
        missing = [str(idx) for idx, count in enumerate(counts) if count == 0]
        raise ValueError(f"Cannot compute class weights. Missing class index(es): {', '.join(missing)}")

    frequencies = counts / counts.sum()
    if strategy == "inverse_frequency":
        raw_weights = 1.0 / frequencies
    elif strategy == "sqrt_inverse":
        raw_weights = 1.0 / np.sqrt(frequencies)
    else:
        raise ValueError(f"Unsupported class weight strategy: {strategy}")

    weights = raw_weights / raw_weights.mean()
    weights = np.clip(weights, 1.0 / max_weight, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


def estimate_class_weights(mask_paths, num_classes=3, max_weight=5.0, strategy="inverse_frequency"):
    counts = np.zeros(num_classes, dtype=np.float64)
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask patch for class weighting: {mask_path}")
        values = mask.reshape(-1)
        values = values[(values >= 0) & (values < num_classes)]
        counts += np.bincount(values, minlength=num_classes)[:num_classes]

    return weights_from_counts(counts, max_weight=max_weight, strategy=strategy), counts


def foreground_weights_from_class_counts(class_counts, max_weight=5.0, strategy="inverse_frequency"):
    class_counts = np.asarray(class_counts, dtype=np.float64)
    if class_counts.shape[0] < 2:
        raise ValueError("foreground weighting requires at least background and foreground counts.")
    foreground_counts = np.array([class_counts[0], class_counts[1:].sum()], dtype=np.float64)
    return weights_from_counts(foreground_counts, max_weight=max_weight, strategy=strategy)


def create_loss(
    loss_name,
    class_weights=None,
    foreground_class_weights=None,
    ce_weight=1.0,
    dice_weight=1.0,
    foreground_aux_weight=0.3,
):
    if loss_name == "dice":
        return lambda predictions, targets: kornia.losses.dice_loss(
            get_segmentation_logits(predictions),
            targets,
        )
    if loss_name == "ce":
        cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        return lambda predictions, targets: cross_entropy(
            get_segmentation_logits(predictions),
            targets,
        )
    if loss_name == "ce_dice":
        return CrossEntropyDiceLoss(
            class_weights=class_weights,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        )
    if loss_name == "ce_dice_aux_foreground":
        return AuxiliaryForegroundLoss(
            CrossEntropyDiceLoss(
                class_weights=class_weights,
                ce_weight=ce_weight,
                dice_weight=dice_weight,
            ),
            foreground_aux_weight=foreground_aux_weight,
            foreground_class_weights=foreground_class_weights,
        )
    raise ValueError(f"Unsupported loss_name: {loss_name}")
