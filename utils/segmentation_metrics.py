import numpy as np


def confusion_matrix_from_arrays(gt_arrays, pred_arrays, num_classes=3):
    """Build a pixel-level confusion matrix with rows=true labels and columns=predicted labels."""
    if len(gt_arrays) != len(pred_arrays):
        raise ValueError(
            f"Ground-truth/prediction count mismatch: {len(gt_arrays)} vs {len(pred_arrays)}"
        )

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pred in zip(gt_arrays, pred_arrays):
        if gt.shape != pred.shape:
            raise ValueError(f"Ground-truth/prediction shape mismatch: {gt.shape} vs {pred.shape}")
        valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
        encoded = (gt[valid].astype(np.int64) * num_classes) + pred[valid].astype(np.int64)
        confusion += np.bincount(
            encoded.reshape(-1),
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    return confusion


def metrics_from_confusion(confusion, class_names=None, eps=1e-8):
    """Return segmentation metrics derived from a pixel-level confusion matrix."""
    confusion = np.asarray(confusion, dtype=np.float64)
    num_classes = confusion.shape[0]
    if class_names is None:
        class_names = [str(index) for index in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError("class_names length must match confusion matrix size.")

    true_positive = np.diag(confusion)
    predicted_positive = confusion.sum(axis=0)
    actual_positive = confusion.sum(axis=1)
    total = confusion.sum()

    precision = true_positive / np.maximum(predicted_positive, eps)
    recall = true_positive / np.maximum(actual_positive, eps)
    f1_score = (2 * precision * recall) / np.maximum(precision + recall, eps)
    union = actual_positive + predicted_positive - true_positive
    iou = true_positive / np.maximum(union, eps)
    dice = (2 * true_positive) / np.maximum(actual_positive + predicted_positive, eps)

    present = actual_positive > 0
    mean_iou = float(np.mean(iou[present])) if np.any(present) else 0.0
    mean_dice = float(np.mean(dice[present])) if np.any(present) else 0.0
    pixel_accuracy = float(true_positive.sum() / total) if total else 0.0

    per_class = []
    for index, name in enumerate(class_names):
        per_class.append(
            {
                "class_id": index,
                "class": name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1_score[index]),
                "iou": float(iou[index]),
                "dice": float(dice[index]),
                "support": int(actual_positive[index]),
            }
        )

    return {
        "pixel_accuracy": pixel_accuracy,
        "global_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "per_class": per_class,
        "support": int(total),
    }


def flatten_metrics_for_summary(model_name, metrics):
    """Convert metrics_from_confusion output into report rows."""
    rows = []
    for item in metrics["per_class"]:
        row = dict(item)
        row["model"] = model_name
        row["row_type"] = "class"
        row["pixel_accuracy"] = metrics["pixel_accuracy"]
        row["global_accuracy"] = metrics["global_accuracy"]
        row["mean_iou"] = metrics["mean_iou"]
        row["mean_dice"] = metrics["mean_dice"]
        rows.append(row)

    rows.append(
        {
            "model": model_name,
            "row_type": "overall",
            "class_id": "",
            "class": "overall",
            "precision": "",
            "recall": "",
            "f1_score": "",
            "iou": "",
            "dice": "",
            "support": metrics["support"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "global_accuracy": metrics["global_accuracy"],
            "mean_iou": metrics["mean_iou"],
            "mean_dice": metrics["mean_dice"],
        }
    )
    return rows
