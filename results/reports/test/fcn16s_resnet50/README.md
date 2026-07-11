# Prediction Report: `test`

This report compares prediction masks against the ground-truth masks in the dataset.
A one-model report is a checkpoint-vs-ground-truth evaluation. A multi-model report uses the same ground truth and compares each listed prediction directory independently.

## Dataset

- Root path: `.`
- Subset: `test`
- Number of evaluated raw captures: 3

## Models

- `fcn16s_resnet50`
  - checkpoint/source: `models\fcn16s_resnet50_dil0_bilin1_pre1.pth.tar`
  - prediction directory: `results\predictions\test\fcn16s_resnet50`

## Label Mapping

| Label | Class | Visualization color |
| --- | --- | --- |
| BG | Background/soil | gray |
| S | Sorghum | blue |
| W | Weed | orange |

## Outputs

- `metrics_summary.csv`: pixel-level precision, recall, F1, IoU, Dice, mIoU, mean Dice, support, and global accuracy.
- `metrics_summary.json`: JSON copy of the pixel-level summary metrics.
- `metrics_per_image.csv`: per-capture pixel accuracy, mIoU, Dice, precision, recall, and F1 scores.
- `confusion_matrix_<model>.csv`: raw confusion matrix counts.
- `confusion_matrix_<model>.png`: normalized confusion matrix.
- `qualitative_grid.png`: raw patch, ground truth, prediction, overlay, and error map visualization.

`qualitative_grid.png` uses the raw UAV crop as input context, the dataset mask as ground truth, each model mask as prediction, an overlay for visual inspection, and an error map for failure analysis.
Error map colors: black = correct, red = false-positive foreground, blue = false-negative foreground, yellow = wrong foreground class.

## Representative Crop Coordinates

- 1: image_index=1, x=3600, y=1000
- 2: image_index=2, x=1000, y=2800
- 3: image_index=1, x=5000, y=200
- 4: image_index=0, x=3000, y=800
- 5: image_index=0, x=200, y=800
- 6: image_index=2, x=5000, y=1600
