# AGENTS.md

Operational guidance for AI agents working in this repository.

## Primary Goal

Keep the UAV weed segmentation workflow runnable on the local Windows machine with:

```text
Python launcher: py
Python target:   3.13.12
Venv path:       .venv/
GPU target:      NVIDIA RTX 5060 8 GB
Primary command: .\.venv\Scripts\python.exe
```

Do not rewrite training or model logic unless runtime evidence shows it is required. Prefer small, traceable changes that keep the original model behavior intact.

## Current Model Surface

Research framing as of 2026-07-10:

```text
Proposed v1:  MobileNetV3_Large U-Net with PPM skip/context module and SE/H-swish decoder
Proposed v2:  Proposed v1 plus an auxiliary BG-vs-vegetation foreground head
Baseline set: existing ResNet-backed fcn8s, fcn16s, fcn32s, unet, and dlplus models
```

The proposed v1 model is implemented as `unet_mobilenetv3 mobilenetv3_large`.
The proposed v2 model is implemented as `unet_mobilenetv3_aux mobilenetv3_large`.
Both are documented in `docs/PROPOSED_MODEL_UNET_MOBILENETV3.md`. Do not claim either proposed variant is trained or better than a baseline unless a checkpoint, prediction report, and comparison report exist locally.

Current final-candidate proposed model for ResNet-50 comparison work:

```text
architecture = unet_mobilenetv3_aux
encoder      = mobilenetv3_large
checkpoint   = models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
```

This candidate is not a license to overclaim. Report it as near-top ResNet-50 segmentation quality with a much smaller MobileNetV3-based model unless a newer verified report changes that conclusion.
The old ResNet-34/ResNet-50 baseline artifacts were intentionally cleaned on 2026-07-10 for fair retraining. Do not claim the baseline comparison is current until the baseline checkpoints and reports have been regenerated.

Proposed-model ablation variants are trainable/evaluable as:

```text
unet_mobilenetv3_base mobilenetv3_large
unet_mobilenetv3_ppm mobilenetv3_large
unet_mobilenetv3 mobilenetv3_large
unet_mobilenetv3_aux mobilenetv3_large
```

Interpretation:

```text
unet_mobilenetv3_base = MobileNetV3 U-Net without PPM and without SE decoder
unet_mobilenetv3_ppm  = MobileNetV3 U-Net with PPM and without SE decoder
unet_mobilenetv3      = proposed MobileNetV3 U-Net with PPM and SE/H-swish decoder
unet_mobilenetv3_aux  = proposed v2 with PPM, SE/H-swish decoder, and auxiliary foreground head
```

The baseline model matrix is:

```text
architectures = fcn8s, fcn16s, fcn32s, unet, dlplus
encoders      = resnet18, resnet34, resnet50, resnet101
```

Total baseline combinations:

```text
5 architectures x 4 encoders = 20 base models
```

The proposed models are separate non-ResNet targets:

```text
proposed v1 architecture = unet_mobilenetv3
proposed v2 architecture = unet_mobilenetv3_aux
encoder                   = mobilenetv3_large
```

For architecture comparison, keep the baseline encoder fixed and compare the five baseline architectures plus one proposed model variant. The current final-candidate workflow uses the existing ResNet-50 baselines plus proposed v2:

```text
fcn8s_resnet50
fcn16s_resnet50
fcn32s_resnet50
unet_resnet50
dlplus_resnet50
unet_mobilenetv3_aux
```

ResNet-34 remains useful for quick progress or historical v1 reproduction. Do not present ResNet-34 as the current final comparison unless the user explicitly switches the target encoder.

Encoder-specific comparison scripts are independent files:

```text
scripts/run_proposed_model.ps1
scripts/run_resnet18_architecture_comparison.ps1
scripts/run_resnet34_architecture_comparison.ps1
scripts/run_resnet50_architecture_comparison.ps1
scripts/run_resnet101_architecture_comparison.ps1
```

Shared implementation:

```text
scripts/run_architecture_comparison_core.ps1
scripts/run_model_suite.py
evaluate_model_suite.py
utils/model_registry.py
utils/labels.py
```

Do not duplicate the core training/prediction loop into each encoder wrapper. The PowerShell core delegates to `scripts/run_model_suite.py`; Python-side model selection comes from `utils/model_registry.py`.
Do not duplicate Python-side model lists or class label/color definitions. Use `utils/model_registry.py` for model matrix definitions and `utils/labels.py` for class names/colors.
Do not collapse `unet_mobilenetv3` and `unet_mobilenetv3_aux` into one checkpoint name. They are separate restore points.

For edge/export work, update all of these in the same change set when behavior, commands, or evidence changes:

```text
README.md
docs/DEVELOPMENT_TRACKER.md
docs/EDGE_NCNN_RASPBERRY_PI5.md
docs/EDGE_INFERENCE_IMPLEMENTATION_NOTES.md
```

The export target is `unet_mobilenetv3_aux mobilenetv3_large`. The deployment output is the 3-class segmentation logits only. Do not export the raw dict output, do not switch to proposed v1 silently, and do not treat the auxiliary foreground head as the final classifier.

Use `scripts/check_webcam_inference.py` for local OpenCV camera/video/image sanity checks. Treat it as a local inference contract check, not as Raspberry Pi 5 deployment proof. A webcam run may use `--mode pytorch --view gui` first; `--mode ncnn` requires a compatible Python `ncnn` binding. Keep the GUI watermark text as `code by gedxxe` unless the user explicitly changes it.

## Evidence Rules

Do not claim a model is trained unless its checkpoint exists under `models/`.

Do not claim a model was evaluated unless all of these exist:

```text
results/predictions/<subset>/<model_name>/*_pred.png
results/reports/<subset>/<model_name>/metrics_summary.csv
results/reports/<subset>/<model_name>/qualitative_grid.png
```

Do not claim a multi-model comparison exists unless this exists:

```text
results/reports/<subset>/architecture_comparison_<encoder_name>/metrics_summary.csv
```

Do not claim complete results-and-discussion evaluation exists unless this exists:

```text
results/reports/<subset>/<mode>_evaluation_<encoder_name>/evaluation_summary.csv
results/reports/<subset>/<mode>_evaluation_<encoder_name>/evaluation_summary.json
results/reports/<subset>/<mode>_evaluation_<encoder_name>/evaluation_summary.md
```

For proposed-only evaluation, the directory is:

```text
results/reports/<subset>/proposed_evaluation/
```

For proposed-v2-only evaluation, the directory is:

```text
results/reports/<subset>/proposed_v2_evaluation/
```

Do not claim efficiency results exist unless this exists in that same evaluation report:

```text
efficiency_metrics.csv
efficiency_metrics.json
```

Do not claim NCNN export success for the edge/Raspberry Pi 5 workflow unless all of these exist:

```text
exports/ncnn/unet_mobilenetv3_aux_256/model.ncnn.param
exports/ncnn/unet_mobilenetv3_aux_256/model.ncnn.bin
exports/ncnn/unet_mobilenetv3_aux_256/export_manifest.json
exports/ncnn/unet_mobilenetv3_aux_256/parity_report.json
```

and `export_manifest.json` records:

```text
artifacts.ncnn_exported = true
```

If only `model.pt`, manifest, and parity report exist, describe the state as a partial TorchScript export with NCNN conversion blocked or pending. Do not claim Raspberry Pi 5 real-time performance until a Pi-side runtime command and timing report exist.

Do not claim local webcam inference was checked unless `scripts/check_webcam_inference.py` has processed at least one frame or still image and written a manifest under:

```text
results/webcam_inference_checks/<run_name>/manifest.json
```

When reporting progress, include:

- exact command or script used
- model name
- checkpoint path
- prediction directory
- report directory
- whether the run is quick-validation/progress-grade or final-grade

## Dataset Contract

Raw dataset layout:

```text
data/trainval/img/*.jpg
data/trainval/msk/*.png
data/test/img/*.jpg
data/test/msk/*.png
data/test_different_bbch/img/*.jpg
data/test_different_bbch/msk/*.png
```

Reporting pairs raw images and masks by exact stem first. If stems differ, it also accepts normalized suffix pairs such as:

```text
bbch15_img.jpg <-> bbch15_msk.png
bbch19_img.jpg <-> bbch19_msk.png
```

Only the suffixes `_img`, `_image`, `_msk`, and `_mask` are normalized. Any other mismatch must fail explicitly instead of silently pairing the wrong files.

Patch layout used by training:

```text
data/<subset>/patches/img/*.png
data/<subset>/patches/msk/*.png
```

Run this before training if patch folders are missing:

```powershell
.\.venv\Scripts\python.exe save_patches.py --root_path .
.\.venv\Scripts\python.exe scripts\check_dataset.py --require-patches
```

## Comparison Contract

Use `compare_model_predictions.py` for direct comparison of saved prediction folders. It does not train and does not run inference. It compares saved prediction masks against the ground-truth masks.

Correct comparison structure:

```powershell
.\.venv\Scripts\python.exe compare_model_predictions.py test --root_path . --prediction model_a=results\predictions\test\model_a --prediction model_b=results\predictions\test\model_b --report_dir results\reports\test\comparison_name
```

For the current final-candidate six-model comparison after a cleanup/reset, use the proposed-v2 ResNet-50 wrapper with `-SkipProposedTraining`. This keeps the proposed-v2 checkpoint fixed and retrains/evaluates the five ResNet-50 baselines:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v2 -SkipProposedTraining -NFolds 2 -NTrials 2 -MaxEpochs 150 -BatchSize 4 -NumWorkers 2 -RunPrefix fair_resnet50_v2 -EarlyStopPatience 60 -LrSchedulerPatience 10
```

That script builds a six-model report: five ResNet-50 baselines plus `unet_mobilenetv3_aux`. Use `-SkipTraining` only after all required checkpoints exist and the goal is report regeneration. Use `-BaselineOnly` only when a five-baseline report is intentionally required. Use `scripts/run_proposed_model.ps1 -Variant v2` or `scripts/run_model_suite.py proposed_v2` for proposed-v2-only runs.

For historical v1 ResNet-34 progress reproduction, the old wrapper command is still available:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -BatchSize 8 -NumWorkers 2 -RunPrefix archcmp_resnet34
```

The proposed model checkpoint name is shared across encoder wrappers:

```text
models/unet_mobilenetv3_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
```

The proposed v2 checkpoint is separate:

```text
models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
```

The ResNet-50 baseline checkpoint names are:

```text
models/fcn8s_resnet50_dil0_bilin1_pre1.pth.tar
models/fcn16s_resnet50_dil0_bilin1_pre1.pth.tar
models/fcn32s_resnet50_dil0_bilin1_pre1.pth.tar
models/unet_resnet50_dil0_bilin1_pre1.pth.tar
models/dlplus_resnet50_dil0_bilin1_pre1.pth.tar
```

As of the 2026-07-10 cleanup, those baseline checkpoints may be absent by design and must be recreated for a fair new comparison. Do not delete a newly trained baseline checkpoint unless the user explicitly asks for another reset.

Running multiple encoder wrappers can retrain and overwrite the v1 proposed checkpoint. For the current v2 workflow, keep `-ProposedVariant v2` explicit. Use `-SkipProposedTraining` when retraining baselines against the fixed proposed-v2 checkpoint, and use `-SkipTraining` only when the goal is report regeneration from existing checkpoints. Use `-BaselineOnly` only when the proposed model should not be included.

Use `-PlanOnly` before a long run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -PlanOnly
```

Script parameter policy:

- `NFolds`: cross-validation split count; use `2` for progress and `4` for final-style comparison.
- `MaxEpochs`: maximum epochs per fold; use `10` for quick progress, `20` for stronger progress, and `100` for final-style runs.
- `EarlyStopPatience`: epochs without validation improvement before stopping a fold; increase it when increasing `MaxEpochs`.
- `LrSchedulerPatience`: epochs before `ReduceLROnPlateau` lowers LR; keep conservative unless intentionally tuning long runs.
- `NTrials`: Optuna trial count; keep `1` for architecture comparison unless every model receives the same hyperparameter-search budget.
- `BatchSize`: VRAM control; use `8` for ResNet-18/34, `4` for ResNet-50, and `2` for ResNet-101 on RTX 5060 8 GB.
- `NumWorkers`: training DataLoader workers; use `2` normally and `0` if Windows multiprocessing fails.
- `Device`: runtime device; use `cuda` for the RTX 5060 run, `auto` for portable checks, and `cpu` only for quick validation/debug.
- `RunPrefix`: experiment namespace for Optuna DB names; change it for new batches.
- `CleanStudy`: only use for intentional reruns that should delete existing trials in the same study.
- `NoPretrained`: only use when pretrained weight download/cache fails; do not mix pretrained and non-pretrained runs in one comparison.
- `ProposedVariant`: `v1` uses `unet_mobilenetv3`; `v2` uses `unet_mobilenetv3_aux`.
- `BaselineOnly`: skips `unet_mobilenetv3` and runs only the five baseline models.
- `SkipProposedTraining`: keeps the selected proposed variant in prediction/report comparison but does not retrain its checkpoint.
- `SkipEvaluation`: skips the final complete evaluation summary after prediction/comparison.
- `SkipEfficiency`: keeps segmentation evaluation but skips Params/GFLOPs/FPS/latency benchmarking.
- `EvaluationInputSize`: benchmark input size; default is `480`.
- `BenchmarkIterations`: timed benchmark iterations for FPS/latency.
- `CpuLatency`: also measures CPU latency; this can be slow.
- `ClassWeightStrategy`: class-weight formula when automatic class weights are enabled. Default `inverse_frequency` preserves the existing behavior; `sqrt_inverse` is less aggressive and is useful when the proposed model over-favors the rarer Weed class at Sorghum's expense.
- `ValidationLoss`: checkpoint selection loss for baseline models; default `dice` preserves the original baseline behavior.
- `ProposedValidationLoss`: checkpoint selection objective for the proposed model; default `macro_f1` selects the proposed checkpoint by validation macro F1 while training still uses the class-aware loss. `foreground_macro_f1` averages Sorghum and Weed F1 only.
- `ForegroundAuxWeight`: auxiliary BG-vs-vegetation CE loss weight for `unet_mobilenetv3_aux`; default `0.3`.
- For `NTrials > 1`, `train.py` saves per-trial checkpoints and publishes the best available trial checkpoint to the canonical model path after Optuna finishes. Use a fresh `RunPrefix` for materially different experiments.

DeepLabV3+ training can fail if the final training batch has exactly one patch because the ASPP pooling branch applies BatchNorm to a 1x1 tensor. The shared `get_loaders()` function intentionally drops only that final singleton training batch. Do not remove this guard unless the ASPP normalization is redesigned and verified.

Prediction image reconstruction derives patch grid dimensions from `mask_shape` in `utils/predict.py`. Do not reintroduce raw-image-size branches for specific resolutions unless there is runtime evidence that a dataset requires a special layout.
Prediction post-processing for multi-class models must use `argmax` over segmentation logits. Do not reintroduce sigmoid-threshold post-processing for the three-class BG/Sorghum/Weed problem.

## Verification Checklist

Before handing results back to the user, run the checks that match the change:

```powershell
.\.venv\Scripts\python.exe -m py_compile train.py predict_testset.py compare_model_predictions.py evaluate_model_suite.py utils\losses.py utils\parser.py utils\reporting.py utils\model_outputs.py utils\manual_unet_mobilenetv3.py scripts\run_model_suite.py scripts\check_comparison_status.py
```

For report-only changes after prediction folders exist:

```powershell
.\.venv\Scripts\python.exe compare_model_predictions.py test --root_path . --prediction current=results\predictions\test --report_dir results\reports\test\current_check --max_examples 1
```

If `results/predictions/` has been cleaned, regenerate predictions first or limit validation to `-PlanOnly` script checks.

For CLI/script changes:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet18_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 1 -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 1 -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 1 -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet101_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 1 -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -Variant v2 -NFolds 2 -NTrials 2 -MaxEpochs 150 -EarlyStopPatience 60 -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v2 -NFolds 2 -NTrials 2 -MaxEpochs 150 -EarlyStopPatience 60 -SkipTraining -PlanOnly
```

For fair baseline reset cleanup, use dry-run first:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\cleanup_retrain_baselines.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\cleanup_retrain_baselines.ps1 -Apply
```

This cleanup removes selected ResNet-34/ResNet-50 baseline artifacts and old proposed-v1 artifacts while preserving `.venv`, `data`, `exports`, `results/training_logs`, and the proposed-v2 checkpoint.

For status reporting, use the checker instead of manually inferring completion:

```powershell
.\.venv\Scripts\python.exe scripts\check_comparison_status.py --root_path . --subset test --write docs\COMPARISON_STATUS.md
.\.venv\Scripts\python.exe scripts\check_comparison_status.py --root_path . --subset test --proposed_variant v2 --write docs\COMPARISON_STATUS_V2.md
```

For complete evaluation pipeline checks:

```powershell
.\.venv\Scripts\python.exe evaluate_model_suite.py full --encoder resnet34 --root_path . --subset test --skip_efficiency --report_dir results\reports\test\manual_verification_resnet34 --max_examples 1
.\.venv\Scripts\python.exe evaluate_model_suite.py proposed --root_path . --subset test --device cuda --warmup_iterations 0 --benchmark_iterations 1 --report_dir results\reports\test\manual_verification_proposed_eff --max_examples 1
```

For conservative cleanup, use dry-run first:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\cleanup_workspace.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\cleanup_workspace.ps1 -Apply
```

The cleanup script must remain conservative. It may remove repo-level caches and known orphan prediction files, but it must not remove `.venv`, `data`, `models`, `results/training_logs`, or final report folders.
It may also remove `tmp/` and temporary report folders named `audit_check_*` or `evaluation_check_*`.

## Development Tracker

Keep the current status in:

```text
docs/DEVELOPMENT_TRACKER.md
```

Update the tracker when:

- a model checkpoint is created
- predictions are generated
- comparison reports are generated
- a command fails and requires a workaround
- documentation changes the recommended workflow

If something was not run, mark it as not run. Do not infer completion from intended commands.
