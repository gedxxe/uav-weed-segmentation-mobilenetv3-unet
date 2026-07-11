# Model Comparison Runbook

Tujuan dokumen ini: menjalankan eksperimen dengan mode yang eksplisit.

| Mode | Isi eksperimen | Kapan dipakai |
| --- | --- | --- |
| `baseline` | 5 baseline: `fcn8s`, `fcn16s`, `fcn32s`, `unet`, `dlplus` dengan satu encoder ResNet yang sama. | Membandingkan arsitektur baseline tanpa proposed model. |
| `proposed` | 1 proposed v1 model: `unet_mobilenetv3 mobilenetv3_large`. | Melatih atau mengevaluasi proposed v1 saja. |
| `proposed_v2` | 1 proposed v2 model: `unet_mobilenetv3_aux mobilenetv3_large`. | Melatih atau mengevaluasi model segmentasi v2; auxiliary foreground head hanya dipakai sebagai training supervision. |
| `full` | 6 model: 5 baseline fixed-ResNet plus proposed v1. | Komparasi utama v1. |
| `full_v2` | 6 model: 5 baseline fixed-ResNet plus proposed v2. | Komparasi utama v2 tanpa menimpa report v1. |
| `ablation` | `unet_<encoder>`, `unet_mobilenetv3_base`, `unet_mobilenetv3_ppm`, `unet_mobilenetv3`, `unet_mobilenetv3_aux`. | Ablation study proposed model. |

## 1. Model Surface

Baseline memakai kombinasi:

```text
architectures = fcn8s, fcn16s, fcn32s, unet, dlplus
encoders      = resnet18, resnet34, resnet50, resnet101
```

Proposed models terpisah:

```text
proposed v1 architecture = unet_mobilenetv3
proposed v2 architecture = unet_mobilenetv3_aux
encoder                   = mobilenetv3_large
```

Jangan menyebut `unet_mobilenetv3` sebagai encoder ResNet. Itu arsitektur proposed yang punya backbone MobileNetV3_Large.

Default training recipe by model family:

```text
baseline models:     dice loss, no class weighting
proposed v1 model:   ce_dice loss, automatic class weights, validation_loss=macro_f1
proposed v2 model:   ce_dice_aux_foreground loss, automatic class weights, validation_loss=foreground_macro_f1 for the current final-candidate command
```

Reason: the first proposed-model failure mode was mostly `Weed -> Sorghum` confusion, not foreground/background failure. `ce_dice` keeps segmentation overlap pressure from Dice while adding class-discriminative CE pressure for minority classes. `validation_loss=macro_f1` selects the checkpoint by the same class-balanced metric family used in the reports, instead of selecting only by the old Dice validation loss.

Important naming note: `foreground_macro_f1` is computed from the main 3-class segmentation logits by averaging F1 over Sorghum and Weed. It is not computed from the auxiliary binary foreground head.

The older post-ResNet-50 v1 tuning pass targeted balanced Sorghum+Weed foreground performance with `ClassWeightStrategy sqrt_inverse` and `ValidationLoss foreground_macro_f1`. Keep that recipe as historical unless intentionally rerunning the v1 tuning branch.

## 2. Pilih Encoder Baseline

Gunakan satu encoder ResNet untuk baseline dalam satu comparison.

| Encoder baseline | Batch awal RTX 5060 8 GB | Kapan dipakai |
| --- | ---: | --- |
| `resnet18` | 8 | Cepat, baseline ringan. |
| `resnet34` | 8 | Cepat untuk progress, ablation, atau reproduksi v1 historis. |
| `resnet50` | 4 | Jalur pembanding utama saat ini untuk proposed v2. |
| `resnet101` | 2 | Paling berat, bukan prioritas progress cepat. |

## 3. PowerShell Commands

### Baseline Only

Preview:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -BaselineOnly -PlanOnly
```

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -BatchSize 8 -NumWorkers 2 -RunPrefix baseline_resnet34 -BaselineOnly
```

### Proposed Only

Preview:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -NFolds 2 -MaxEpochs 10 -PlanOnly
```

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -NFolds 2 -MaxEpochs 10 -BatchSize 8 -NumWorkers 2 -RunPrefix proposed_mnv3
```

### Proposed V2 Only

Preview:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -Variant v2 -NFolds 2 -NTrials 2 -MaxEpochs 150 -EarlyStopPatience 60 -PlanOnly
```

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -Variant v2 -NFolds 2 -NTrials 2 -MaxEpochs 150 -BatchSize 8 -NumWorkers 2 -RunPrefix proposed_mnv3_aux_fg_e150 -ClassWeightMax 8 -ClassWeightStrategy inverse_frequency -CeWeight 1.0 -DiceWeight 1.0 -ForegroundAuxWeight 0.3 -ValidationLoss foreground_macro_f1 -EarlyStopPatience 60 -LrSchedulerPatience 10
```

### Historical Full V1 ResNet-34 Comparison

Preview:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -PlanOnly
```

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -BatchSize 8 -NumWorkers 2 -RunPrefix archcmp_resnet34
```

This v1 progress path trains and evaluates:

```text
fcn8s_resnet34
fcn16s_resnet34
fcn32s_resnet34
unet_resnet34
dlplus_resnet34
unet_mobilenetv3
```

### Current Full V2 ResNet-50 Comparison

After proposed v2 training has produced its checkpoint, retrain the five ResNet-50 baselines against the fixed proposed-v2 checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v2 -SkipProposedTraining -NFolds 2 -NTrials 2 -MaxEpochs 150 -BatchSize 4 -NumWorkers 2 -RunPrefix fair_resnet50_v2 -EarlyStopPatience 60 -LrSchedulerPatience 10
```

This uses:

```text
fcn8s_resnet50
fcn16s_resnet50
fcn32s_resnet50
unet_resnet50
dlplus_resnet50
unet_mobilenetv3_aux
```

The PowerShell wrappers now build the complete evaluation summary after prediction/comparison unless `-SkipEvaluation` is passed. Use `-SkipEfficiency` when you only need segmentation metrics and qualitative outputs without Params/GFLOPs/FPS/latency.

Use `-SkipTraining` only when the required ResNet-50 baseline checkpoints already exist and the goal is report regeneration:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v2 -SkipTraining -BatchSize 4 -NumWorkers 2 -RunPrefix fair_resnet50_v2_report
```

### Full V2 ResNet-50 BBCH Shift Evaluation

Run the same trained checkpoints on `test_different_bbch` without retraining:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v2 -SkipTraining -Subset test_different_bbch -BatchSize 4 -NumWorkers 0 -RunPrefix fair_resnet50_v2_bbch
```

This subset uses image and mask names such as `bbch15_img.jpg` and `bbch15_msk.png`. The reporting pipeline pairs those files by normalized stem and still fails explicitly for ambiguous names.

## Edge Export Pointer

Edge/Raspberry Pi 5 export is intentionally documented outside the training comparison commands. Use these files for the NCNN export and inference contract:

```text
docs/EDGE_NCNN_RASPBERRY_PI5.md
docs/EDGE_INFERENCE_IMPLEMENTATION_NOTES.md
```

Do not mix edge export commands into baseline/proposed training runs. The edge path exports the already trained `unet_mobilenetv3_aux mobilenetv3_large` checkpoint through a segmentation-only wrapper.

## 4. Python Commands

The Python suite runner is:

```text
scripts/run_model_suite.py
```

It supports `baseline`, `proposed`, `proposed_v2`, `full`, `full_v2`, and `ablation` modes.

### Baseline Only

Preview:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py baseline --encoder resnet34 --n_folds 2 --max_epochs 10 --plan_only
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py baseline --encoder resnet34 --n_folds 2 --max_epochs 10 --batch_size 8 --num_workers 2 --run_prefix baseline_resnet34
```

### Proposed Only

Preview:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py proposed --n_folds 2 --max_epochs 10 --plan_only
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py proposed --n_folds 2 --max_epochs 10 --batch_size 8 --num_workers 2 --run_prefix proposed_mnv3
```

### Proposed V2 Only

Preview:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py proposed_v2 --n_folds 2 --n_trials 2 --max_epochs 150 --early_stop_patience 60 --plan_only
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py proposed_v2 --n_folds 2 --n_trials 2 --max_epochs 150 --batch_size 8 --num_workers 2 --run_prefix proposed_mnv3_aux_fg_e150 --class_weight_max 8 --proposed_class_weight_strategy inverse_frequency --ce_weight 1.0 --dice_weight 1.0 --foreground_aux_weight 0.3 --proposed_validation_loss foreground_macro_f1 --early_stop_patience 60 --lr_scheduler_patience 10
```

### Historical Full V1 ResNet-34 Comparison

Preview:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py full --encoder resnet34 --n_folds 2 --max_epochs 10 --plan_only
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py full --encoder resnet34 --n_folds 2 --max_epochs 10 --batch_size 8 --num_workers 2 --run_prefix archcmp_resnet34
```

### Ablation Study

Preview:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py ablation --encoder resnet34 --n_folds 2 --max_epochs 10 --plan_only
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py ablation --encoder resnet34 --n_folds 2 --max_epochs 20 --batch_size 8 --num_workers 2 --run_prefix ablation_mnv3_resnet34
```

Ablation mode compares:

```text
unet_resnet34
unet_mobilenetv3_base
unet_mobilenetv3_ppm
unet_mobilenetv3
unet_mobilenetv3_aux
```

For a stricter ablation where loss function and checkpoint selection are identical across the U-Net row and MobileNetV3 rows, pass the same loss settings explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py ablation --encoder resnet34 --n_folds 4 --max_epochs 100 --batch_size 8 --num_workers 2 --run_prefix final_ablation_mnv3_resnet34 --loss ce_dice --class_weights auto --validation_loss macro_f1 --proposed_loss ce_dice --proposed_class_weights auto --proposed_validation_loss macro_f1
```

## 5. Parameter Meaning

PowerShell uses single-dash parameters:

```text
-NFolds, -MaxEpochs, -BatchSize, -PlanOnly
```

Python uses double-dash parameters:

```text
--n_folds, --max_epochs, --batch_size, --plan_only
```

| Parameter | Meaning | Practical guidance |
| --- | --- | --- |
| `NFolds` / `--n_folds` | Cross-validation split count. | `2` for progress, `4` for final-style. |
| `MaxEpochs` / `--max_epochs` | Maximum epoch count per fold. Early stopping may stop earlier. | `10` quick, `20` stronger progress, `100` final-style. |
| `EarlyStopPatience` / `--early_stop_patience` | Stop a fold after this many epochs without validation improvement. | Increase with long runs; e.g. `25` when `MaxEpochs` is around `120`. |
| `LrSchedulerPatience` / `--lr_scheduler_patience` | Epoch patience before reducing LR on validation plateau. | Keep `5` normally; use `8-10` only for long slow runs. |
| `NTrials` / `--n_trials` | Optuna trial count. | Keep `1` unless every model gets the same larger search budget. |
| `BatchSize` / `--batch_size` | Patch count per training batch. | RTX 5060 starting point: `8` for ResNet-18/34/proposed, `4` for ResNet-50, `2` for ResNet-101. |
| `NumWorkers` / `--num_workers` | Training DataLoader workers. | `2` normally, `0` if Windows multiprocessing fails. |
| `Device` / `--device` | Runtime device. | Use `cuda` for the RTX 5060 run, `auto` for portable checks, and `cpu` only for quick validation/debug. |
| `RunPrefix` / `--run_prefix` | Prefix for Optuna DB files. | Use a unique value per experiment batch. |
| `CleanStudy` / `--clean_study` | Deletes the existing matching Optuna study before training. | Use only for intentional clean reruns. |
| `NoPretrained` / `--no-pretrained` | Disables ImageNet pretrained weights. | Use only when pretrained download/cache is unavailable. |
| `Loss` / `--loss` | Loss for baseline models. | Default `dice` preserves old baseline behavior. |
| `ClassWeights` / `--class_weights` | Class weighting mode for baseline models. | Default `none`. |
| `Loss` in `run_proposed_model.ps1` / `--proposed_loss` | Loss for proposed model. | Default `ce_dice`. |
| `Variant` / `--mode proposed_v2` | Chooses proposed v1 or v2. | Use PowerShell `-Variant v2` or Python mode `proposed_v2/full_v2` for the auxiliary foreground model. |
| `ForegroundAuxWeight` / `--foreground_aux_weight` | Weight for the v2 auxiliary BG-vs-vegetation CE loss. | Default `0.3`; tune only after a controlled comparison. |
| `ClassWeights` in `run_proposed_model.ps1` / `--proposed_class_weights` | Class weighting mode for proposed model. | Default `auto`. |
| `ClassWeightStrategy` / `--class_weight_strategy` | Formula for automatic baseline class weights. | Default `inverse_frequency`; use `sqrt_inverse` for less aggressive imbalance weighting. |
| `ClassWeightStrategy` in `run_proposed_model.ps1` / `--proposed_class_weight_strategy` | Formula for automatic proposed-model class weights. | Use the recipe being evaluated; the current proposed-v2 final-candidate run used `inverse_frequency`. |
| `ValidationLoss` / `--validation_loss` | Checkpoint selection objective for baseline models. | Default `dice` preserves old baseline behavior. Other options: `same`, `macro_f1`, `weed_f1`, `foreground_macro_f1`. |
| `ValidationLoss` in `run_proposed_model.ps1` / `--proposed_validation_loss` | Checkpoint selection objective for proposed model. | Default `macro_f1`; use `foreground_macro_f1` to select by Sorghum+Weed F1 from the main segmentation head only. |
| `SkipTraining` / `--skip_training` | Uses existing checkpoints and skips training. | Useful for regenerating predictions/reports. |
| `SkipPrediction` / `--skip_prediction` | Trains only and skips prediction/report generation. | Useful when evaluation will be run later. |
| `SkipEvaluation` / `--skip_evaluation` | Skips the final complete evaluator. | Use when you only want training/prediction/comparison. |
| `SkipEfficiency` / `--skip_efficiency` | Skips Params/GFLOPs/FPS/latency benchmark but keeps segmentation reports. | Useful for quick report regeneration. |
| `EvaluationInputSize` / `--evaluation_input_size` | Input size used for GFLOPs and latency. | Default `480`, matching the requested evaluation setting. |
| `BenchmarkIterations` / `--benchmark_iterations` | Timed iterations for FPS/latency. | Use `20` for progress and `50+` for stronger report numbers. |
| `CpuLatency` / `--cpu_latency` | Also measures CPU inference latency. | Optional; can be slow on heavy baselines. |
| `PlanOnly` / `--plan_only` | Prints commands without executing them. | Use before every long run. |
| `BaselineOnly` | PowerShell-only switch for encoder wrappers that skips proposed model. | Use for a 5-baseline report. |
| `SkipProposedTraining` / `--skip_proposed_training` | Keeps proposed model in prediction/report but does not retrain it. | Use after proposed checkpoint exists when running other baseline encoders. |

Multi-trial checkpoint rule:

```text
When NTrials > 1, train.py saves per-trial checkpoints first and publishes the best available trial checkpoint to the canonical models/<model>.pth.tar path after Optuna finishes.
Use a new RunPrefix for each materially different experiment, or use CleanStudy when intentionally replacing an existing study.
```

## 6. Output Paths

Baseline checkpoints:

```text
models/<architecture>_<encoder>_dil0_bilin1_pre1.pth.tar
```

Proposed checkpoint:

```text
models/unet_mobilenetv3_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
```

Proposed v2 checkpoint:

```text
models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
```

Prediction folders:

```text
results/predictions/<subset>/<model_name>/
```

Model reports:

```text
results/reports/<subset>/<model_name>/
```

Combined reports:

```text
results/reports/<subset>/baseline_comparison_<encoder>/        # Python baseline mode
results/reports/<subset>/proposed_model/                       # Python proposed mode
results/reports/<subset>/proposed_v2_model/                    # Python proposed_v2 mode
results/reports/<subset>/architecture_comparison_<encoder>/    # full six-model mode
results/reports/<subset>/architecture_comparison_<encoder>_proposed_v2/ # full_v2 six-model mode
results/reports/<subset>/full_evaluation_<encoder>/            # complete full evaluation summary
results/reports/<subset>/full_v2_evaluation_<encoder>/         # complete full_v2 evaluation summary
results/reports/<subset>/ablation_evaluation_<encoder>/        # complete ablation evaluation summary
```

Complete evaluation outputs:

```text
evaluation_summary.csv
evaluation_summary.json
evaluation_summary.md
efficiency_metrics.csv
efficiency_metrics.json
metrics_summary.csv
metrics_summary.json
metrics_per_image.csv
qualitative_grid.png
```

Conservative cleanup:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\cleanup_workspace.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\cleanup_workspace.ps1 -Apply
```

The first command is a dry-run. The script intentionally skips `.venv`, `data`, `models`, training logs, and final report folders. It can remove repo-level caches, `tmp/`, old orphan prediction files, and temporary `audit_check_*` / `evaluation_check_*` report folders.

Training logs for new `train.py` runs:

```text
results/training_logs/<run_id>_training_log.csv
results/training_logs/<run_id>_training_log.json
results/training_logs/<run_id>_train_loss_curve.png
results/training_logs/<run_id>_validation_loss_curve.png
results/training_logs/<run_id>_validation_miou_curve.png
results/training_logs/<run_id>_validation_dice_curve.png
```

## 7. Check Status

Use the status checker instead of manual guessing:

```powershell
.\.venv\Scripts\python.exe scripts\check_comparison_status.py --root_path . --subset test --write docs\COMPARISON_STATUS.md
```

Status file:

```text
docs/COMPARISON_STATUS.md
```

## 8. How To Explain To Supervisor

Progress-grade:

```text
Six-model comparison: five fixed-ResNet-34 baselines plus proposed MobileNetV3 U-Net, 2-fold CV, 1 Optuna trial, max 10 epochs, batch size 8, CUDA AMP enabled.
```

Final-style:

```text
Six-model comparison: five fixed-ResNet-34 baselines plus proposed MobileNetV3 U-Net, 4-fold CV, 1 Optuna trial, max 100 epochs, batch size 8, CUDA AMP enabled.
```

Do not call a run final if it uses `NFolds=2`, `--n_folds 2`, or `MaxEpochs=10`.

## 9. Common Errors

CUDA out of memory:

```text
Lower batch size first.
```

Windows DataLoader multiprocessing error:

```text
Use NumWorkers 0 / --num_workers 0.
```

DeepLabV3+ BatchNorm error:

```text
The final train batch may have one patch. The shared train loader drops only that final singleton batch.
Validation still evaluates every patch.
```

Pretrained download/cache failure:

```text
Use NoPretrained / --no-pretrained for every model in the same comparison.
```

Proposed model predicts plant shape but classifies weeds as sorghum:

```text
Use the proposed default recipe: ce_dice + automatic class weights.
Do not judge proposed quality from a short Dice-only run.
```

Prediction/report mismatch:

```text
Multi-class prediction must use argmax over raw logits. If old reports were generated before this fix, regenerate predictions/reports before drawing conclusions.
```

Historical v1 tuning branch: if `unet_mobilenetv3` is lightweight and strong on `Weed` but loses Sorghum recall/F1 versus `unet_resnet50`, run a balanced foreground proposed pass:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -NFolds 2 -MaxEpochs 40 -BatchSize 8 -NumWorkers 2 -RunPrefix proposed_mnv3_fgmacro_sqrt -ClassWeightMax 8 -ClassWeightStrategy sqrt_inverse -CeWeight 1.2 -DiceWeight 1.0 -ValidationLoss foreground_macro_f1
```

If the previous run already used this same `RunPrefix`, choose a new prefix before changing the training recipe, for example:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -NFolds 4 -NTrials 3 -MaxEpochs 150 -BatchSize 8 -NumWorkers 2 -RunPrefix proposed_mnv3_fgmacro_sqrt_e150 -ClassWeightMax 8 -ClassWeightStrategy sqrt_inverse -CeWeight 1.2 -DiceWeight 1.0 -ValidationLoss foreground_macro_f1 -EarlyStopPatience 60 -LrSchedulerPatience 10
```

Then retrain or regenerate the v1 comparison explicitly. The old v1 checkpoint was cleaned on 2026-07-10, so `-SkipTraining` is valid only after the required v1 and baseline checkpoints exist again.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v1 -NFolds 2 -NTrials 2 -MaxEpochs 150 -BatchSize 4 -NumWorkers 2 -RunPrefix restore_v1_resnet50_argmax
```
