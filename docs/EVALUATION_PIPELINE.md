# Evaluation Pipeline

This document explains the complete evaluation workflow for segmentation quality, computational efficiency, training logs, qualitative outputs, and ablation studies.

## Evaluation Entry Point

Use:

```powershell
.\.venv\Scripts\python.exe evaluate_model_suite.py <mode> --encoder <resnet_encoder> --root_path . --subset test --device cuda
```

Supported modes:

| Mode | Models evaluated | Main use |
| --- | --- | --- |
| `baseline` | `fcn8s`, `fcn16s`, `fcn32s`, `unet`, `dlplus` with one fixed ResNet encoder | Five-baseline comparison. |
| `proposed` | `unet_mobilenetv3 mobilenetv3_large` | Proposed v1-only evaluation. |
| `proposed_v2` | `unet_mobilenetv3_aux mobilenetv3_large` | Proposed v2 semantic-segmentation evaluation; auxiliary foreground is training supervision, not the final evaluation output. |
| `full` | Five fixed-ResNet baselines plus proposed v1 | Main v1 six-model results table. |
| `full_v2` | Five fixed-ResNet baselines plus proposed v2 | Main v2 six-model results table without overwriting v1 reports. |
| `ablation` | `unet_<encoder>`, `unet_mobilenetv3_base`, `unet_mobilenetv3_ppm`, `unet_mobilenetv3`, `unet_mobilenetv3_aux` | Proposed-model component ablation. |

The evaluator does not train and does not generate predictions. It reads existing prediction folders and checkpoints, then writes complete reports.
Prediction folders should be regenerated after the 2026-07-09 argmax-logit post-processing fix before final metrics are cited.

## Full Six-Model Evaluation

After training and prediction folders exist:

```powershell
.\.venv\Scripts\python.exe evaluate_model_suite.py full --encoder resnet34 --root_path . --subset test --device cuda
```

Output directory:

```text
results/reports/test/full_evaluation_resnet34/
```

Useful fast check without efficiency benchmarking:

```powershell
.\.venv\Scripts\python.exe evaluate_model_suite.py full --encoder resnet34 --root_path . --subset test --device cuda --skip_efficiency
```

Enable CPU latency too:

```powershell
.\.venv\Scripts\python.exe evaluate_model_suite.py full --encoder resnet34 --root_path . --subset test --device cuda --cpu_latency
```

CPU latency can be slow for `resnet50`, `resnet101`, and `dlplus`.

## PowerShell Suite Integration

The architecture comparison PowerShell wrappers now run the complete evaluator after prediction/comparison, unless skipped.

Full ResNet-34 run with evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -BatchSize 8 -NumWorkers 2 -RunPrefix archcmp_resnet34
```

Skip only the final evaluation summary:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -SkipEvaluation
```

Keep segmentation evaluation but skip efficiency benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -SkipEfficiency
```

## Python Suite Integration

The Python runner supports the same main modes plus ablation:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py full --encoder resnet34 --n_folds 2 --max_epochs 10 --batch_size 8 --num_workers 2 --run_prefix archcmp_resnet34
```

Proposed v2 plus ResNet-50 baseline comparison after the v2 checkpoint exists and the baselines need fair retraining:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py full_v2 --encoder resnet50 --skip_proposed_training --n_folds 2 --n_trials 2 --max_epochs 150 --batch_size 4 --num_workers 2 --run_prefix fair_resnet50_v2 --early_stop_patience 60 --lr_scheduler_patience 10
```

Use `--skip_training` only when all required checkpoints already exist and the goal is report regeneration.

Ablation run:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py ablation --encoder resnet34 --n_folds 2 --max_epochs 20 --batch_size 8 --num_workers 2 --run_prefix ablation_mnv3_resnet34
```

Ablation mode trains and evaluates:

```text
unet_resnet34
unet_mobilenetv3_base
unet_mobilenetv3_ppm
unet_mobilenetv3
unet_mobilenetv3_aux
```

For a stricter ablation where the loss recipe should be the same across all rows, explicitly pass the same loss settings for baseline and proposed branches:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py ablation --encoder resnet34 --n_folds 4 --max_epochs 100 --batch_size 8 --num_workers 2 --run_prefix final_ablation_mnv3_resnet34 --loss ce_dice --class_weights auto --validation_loss macro_f1 --proposed_loss ce_dice --proposed_class_weights auto --proposed_validation_loss macro_f1
```

## Segmentation Metrics

The evaluator computes pixel-level metrics from saved prediction masks against the dataset masks:

| Metric | Meaning |
| --- | --- |
| Pixel Accuracy / Global Accuracy | `sum(TP_c) / total pixels`. |
| IoU per class | `TP_c / (TP_c + FP_c + FN_c)` for each class. |
| mIoU | Mean IoU over present classes. |
| Dice per class | `2TP_c / (2TP_c + FP_c + FN_c)` for each class. |
| Mean Dice | Mean Dice over present classes. |
| Precision per class | `TP_c / (TP_c + FP_c)` for each class. |
| Recall per class | `TP_c / (TP_c + FN_c)` for each class. |
| F1-score per class | `2PR / (P + R)` for each class. |
| Confusion matrix | Rows are true labels; columns are predicted labels. |

Class order:

```text
0 = Background
1 = Sorghum
2 = Weed
```

For `test_different_bbch`, reporting accepts the dataset's image/mask naming pattern:

```text
bbch15_img.jpg <-> bbch15_msk.png
bbch19_img.jpg <-> bbch19_msk.png
```

The subset is a dataset-shift evaluation. It should be run with `-SkipTraining` when the goal is to test already trained checkpoints.

## Efficiency Metrics

The evaluator reports:

| Metric | Meaning |
| --- | --- |
| Trainable parameters | Parameters with `requires_grad=True`. |
| Total parameters | All parameters. |
| Model size MB | Parameter and buffer memory footprint in MiB. |
| GFLOPs | Approximate Conv/Linear inference FLOPs for the configured input size. Default: 480x480 RGB. |
| FPS | Batch-1 inference throughput. |
| Latency ms/image | Batch-1 inference latency. |
| Peak GPU memory MB | Peak CUDA allocation during benchmark if CUDA is used. |
| CPU latency ms/image | Filled only with `--cpu_latency`. |

FLOP policy:

```text
Conv/Linear multiply-add operations are counted as 2 FLOPs.
```

Use enough benchmark iterations for report numbers:

```powershell
.\.venv\Scripts\python.exe evaluate_model_suite.py full --encoder resnet34 --root_path . --subset test --device cuda --warmup_iterations 10 --benchmark_iterations 50
```

Do not use a one-iteration benchmark as a final FPS/latency number because the first CUDA pass can include setup overhead.

The evaluation summary stores the benchmark input size in `benchmark_input_size` and the measured FLOPs in `gflops`, so the table remains valid if `--input_size` is changed.

## Training Logs

New `train.py` runs write per-epoch logs to:

```text
results/training_logs/
```

Generated files:

```text
<run_id>_training_log.csv
<run_id>_training_log.json
<run_id>_training_summary.json
<run_id>_train_loss_curve.png
<run_id>_validation_loss_curve.png
<run_id>_validation_miou_curve.png
<run_id>_validation_dice_curve.png
<run_id>_learning_rate_curve.png
```

Logged columns include:

```text
trial, fold, epoch, train_loss, valid_loss, validation_mean_iou,
validation_mean_dice, validation_pixel_accuracy, learning_rate,
epoch_time_sec, loss, class_weights, validation_loss_mode
```

Old runs that finished before this logging was added cannot be fully reconstructed per epoch from checkpoints alone.

## Qualitative Outputs

Every report writes:

```text
qualitative_grid.png
```

Rows include:

```text
Original image patch
Ground truth mask
Predicted mask
Prediction overlay on image
Error map
```

Error map colors:

```text
black  = correct
red    = false-positive foreground
blue   = false-negative foreground
yellow = wrong foreground class, for example Weed predicted as Sorghum
```

## Output Files

Each evaluation report writes:

```text
metrics_summary.csv
metrics_summary.json
metrics_per_image.csv
confusion_matrix_<model>.csv
confusion_matrix_<model>.png
qualitative_grid.png
manifest.json
evaluation_summary.csv
evaluation_summary.json
evaluation_summary.md
efficiency_metrics.csv
efficiency_metrics.json
README.md
```

If `--skip_efficiency` is used, `efficiency_metrics.csv/json` are not written and efficiency columns in `evaluation_summary.*` stay empty.

If one model fails during efficiency benchmarking, the evaluator records the failure in the `efficiency_error` column and continues writing the segmentation report and summary tables. This prevents a late FPS/GFLOPs issue from discarding already-generated segmentation results.

## Edge Export Parity

Edge export parity is not the same as the segmentation mIoU report. The mIoU report compares predicted masks against ground truth. Export parity compares two inference backends for the same input.

For the proposed-v2 NCNN path, use:

```text
PyTorch segmentation-only wrapper output
versus
TorchScript / ONNX / NCNN output
```

Acceptance policy:

```text
TorchScript or ONNX max_abs_error on logits <= 1e-3
NCNN argmax mask agreement >= 99.5 percent on a real patch
class order unchanged: Background, Sorghum, Weed
```

Small raw-logit differences can be acceptable when the argmax mask is stable. A large argmax mismatch means the export is not valid, even if `.param` and `.bin` files were produced.

Detailed edge-export documentation:

```text
docs/EDGE_NCNN_RASPBERRY_PI5.md
docs/EDGE_INFERENCE_IMPLEMENTATION_NOTES.md
```
