# Development Tracker

Last updated: 2026-07-11

This file tracks what has actually been implemented or verified in this checkout. Do not use it to record planned work as completed.

## Local Runtime Target

| Item | Status | Evidence |
| --- | --- | --- |
| Python target | Configured | `py -3.13`, expected Python 3.13.12 |
| Virtual environment | Configured | `.venv/` |
| GPU target | Configured | RTX 5060 8 GB CUDA workflow |
| Dataset patch workflow | Implemented | `save_patches.py`, `scripts/check_dataset.py` |
| Windows-safe prediction | Implemented | `predict_testset.py` uses `if __name__ == "__main__"` and prediction defaults to `--num_workers 0` |

## Implemented Improvements

| Area | Status | Notes |
| --- | --- | --- |
| Training CLI | Implemented | Runtime flags for device, AMP, batch size, workers, folds, epochs, and checkpoint saving. |
| RTX 5060 setup docs | Implemented | `README.md` and `docs/TRAINING_WINDOWS_RTX5060.md`. |
| Prediction checkpoint loading | Implemented | Supports `.pt`, `.pth`, and `.pth.tar` style checkpoint paths. |
| Prediction report generation | Implemented | `utils/reporting.py` creates metrics, confusion matrices, qualitative grids, manifest, and report README. |
| BBCH image-mask report pairing | Implemented, unit-tested | `utils/reporting.py` accepts exact stems and normalized `_img`/`_msk` style pairs such as `bbch15_img.jpg` with `bbch15_msk.png`. |
| Multi-model report generation | Implemented | `compare_model_predictions.py` compares saved prediction folders. |
| Complete evaluation pipeline | Implemented, checked | `evaluate_model_suite.py` creates segmentation summary, efficiency summary, JSON/CSV/Markdown exports, and qualitative error-map reports. The old ResNet-34/ResNet-50 baseline reports were intentionally cleaned on 2026-07-10 and must be regenerated before citation. |
| Model registry | Implemented | `utils/model_registry.py` centralizes baseline/proposed architecture lists, encoder lists, default batch sizes, target model specs, checkpoint paths, and prediction/report paths used by Python runners/evaluators. |
| Label registry | Implemented | `utils/labels.py` centralizes class names, short labels, and RGB label colors for patch loading, prediction colorization, and reporting. |
| Dynamic prediction geometry | Implemented, unit-tested | `utils/predict.py` derives the prediction slice grid from `mask_shape` instead of hardcoding raw image resolutions; compatibility checks cover the original dataset sizes and an arbitrary test size. |
| Multi-class prediction post-processing | Implemented, unit-tested | `utils/predict.py` now uses `argmax` over raw segmentation logits for BG/Sorghum/Weed instead of sigmoid-threshold post-processing. Old reports generated before this fix must be regenerated before being cited. |
| Segmentation metric module | Implemented, unit-tested | `utils/segmentation_metrics.py` computes pixel accuracy/global accuracy, IoU, mIoU, Dice, precision, recall, F1, and confusion matrices. |
| Efficiency metric module | Implemented, checked | `utils/efficiency.py` reports params, model size, 480x480 GFLOPs, FPS, latency, peak CUDA memory, and optional CPU latency. Efficiency exports for ResNet-34/ResNet-50 baselines need regeneration after the fair baseline cleanup. |
| Training logs and curves | Implemented for new runs | `train.py` now writes per-epoch CSV/JSON logs and train/validation curve PNGs under `results/training_logs/`. Old finished runs do not have complete epoch logs. |
| Multi-trial checkpoint publishing | Implemented | For `NTrials > 1`, `train.py` saves per-trial checkpoints and publishes the best available trial checkpoint to the canonical model path after Optuna finishes. |
| Suite automation | Implemented | PowerShell and Python entrypoints exist for baseline-only, proposed-only, proposed-v2-only, full six-model, full-v2 six-model, and ablation modes. |
| PowerShell/Python suite consistency | Implemented, checked | Encoder wrappers delegate to `scripts/run_model_suite.py`, so ResNet-18/34/50/101 full comparisons use the same registry-backed model matrix. |
| AI agent project rules | Implemented | `AGENTS.md`. |
| DeepLabV3+ singleton batch guard | Implemented | Training loader drops only a final batch of size 1 to avoid ASPP BatchNorm failure; validation remains complete. |
| Proposed MobileNetV3 U-Net model | Implemented, quick-validated | `utils/manual_unet_mobilenetv3.py`; CPU shape/unit tests passed; local trainable parameter count is 3,980,467; CUDA batch-8 256x256 forward/backward check used about 1390 MiB peak allocation. |
| Proposed ablation variants | Implemented, unit-tested | Trainable architecture names: `unet_mobilenetv3_base`, `unet_mobilenetv3_ppm`, `unet_mobilenetv3`, and `unet_mobilenetv3_aux`. Python suite supports `ablation` mode. |
| Proposed v2 auxiliary foreground model | Implemented, unit-tested | Trainable architecture name: `unet_mobilenetv3_aux`. It keeps MobileNetV3_Large + PPM + SE/H-swish decoder and adds an auxiliary BG-vs-vegetation foreground head. |
| Proposed v2 auxiliary loss | Implemented, unit-tested | `ce_dice_aux_foreground` = 3-class CE+Dice plus `foreground_aux_weight *` binary foreground CE. Auto foreground class weights are derived from the same fold-level mask class counts. |
| Proposed loss recipe | Implemented | Proposed v1 historical tuning used `ce_dice` with automatic class weights and `validation_loss=macro_f1`. The current proposed v2 command uses `ce_dice_aux_foreground`, automatic class weights, `foreground_aux_weight=0.3`, and `validation_loss=foreground_macro_f1`. |
| Proposed foreground tuning controls | Implemented, terminology clarified | Added consecutive early stopping, `class_weight_strategy=sqrt_inverse`, and `foreground_macro_f1` checkpoint selection for Sorghum+Weed balancing without changing architecture. Documentation now states that `foreground_macro_f1` is computed from the main 3-class segmentation head over Sorghum and Weed, not from the auxiliary binary foreground head. |
| Auxiliary-head documentation clarity | Implemented | README, proposed-model notes, runbook, and evaluation pipeline now distinguish the auxiliary BG-vs-vegetation training head from `foreground_macro_f1`, which is a main-head Sorghum+Weed validation objective. |
| CI quick-check workflow | Implemented | `.github/workflows/ci.yml` compiles Python, runs unit tests, and validates PowerShell/Python PlanOnly flows on Windows. |
| Conservative cleanup script | Implemented | `scripts/cleanup_workspace.ps1` is dry-run by default and skips `.venv`, `data`, `models`, training logs, and final reports. It can remove repo caches, `tmp/`, orphan prediction files, and temporary audit/check reports. |
| Fair baseline retrain cleanup script | Implemented, applied | `scripts/cleanup_retrain_baselines.ps1` is dry-run by default. On 2026-07-10 it was applied to remove repo caches/temp files, webcam inference captures, old proposed-v1 artifacts, and old ResNet-34/ResNet-50 baseline checkpoints/predictions/reports/Optuna DBs while preserving `.venv`, `data`, `exports`, `results/training_logs`, and the proposed-v2 checkpoint. |
| README and runbook workflow cleanup | Implemented, re-audited | `README.md`, `AGENTS.md`, `docs/MODEL_COMPARISON_RUNBOOK.md`, and `docs/PROPOSED_MODEL_UNET_MOBILENETV3.md` now present proposed v2 plus ResNet-50 comparison as the primary workflow, keep ResNet-34 as historical/progress context, document model fundamentals, metric formulas, conservative contribution boundaries, docs map, edge export, OpenCV GUI demo, and headless benchmark commands without duplicated command tables. |
| README fair retrain command set | Implemented, PlanOnly checked | `README.md` now lists fair retrain commands for proposed v2 and all baseline encoders: ResNet-18, ResNet-34, ResNet-50, and ResNet-101. The baseline commands use `-SkipProposedTraining` so the proposed-v2 checkpoint remains fixed while baselines retrain. PlanOnly checks passed for proposed v2 and all four baseline encoder wrappers. |
| README patch prerequisite | Implemented, checked | `README.md` now places `save_patches.py --root_path .` and `scripts/check_dataset.py --require-patches` as workflow step 0 before any PowerShell training wrapper. This is meant to prevent missing `data/<subset>/patches/...` path errors on fresh laptops/PCs. Current `check_dataset.py --require-patches` passed with `trainval=2156`, `test=990`, and `test_different_bbch=220` patch image/mask pairs. |
| Edge NCNN export preparation | Implemented, checked locally | `scripts/export_proposed_ncnn.py` exports the proposed v2 checkpoint through a segmentation-only wrapper. Local pnnx CLI produced `model.ncnn.param` and `model.ncnn.bin`; TorchScript parity passed. Pi-side/runtime NCNN parity is still not run. |
| Local OpenCV inference checker | Implemented, checked previously on still image | `scripts/check_webcam_inference.py` supports `pytorch`, `torchscript`, and optional `ncnn` inference backends plus explicit `--view gui` / `--view headless` display modes. Old `results/webcam_inference_checks/` outputs were cleaned on 2026-07-10. Live webcam runs are citeable only after the selected run writes a fresh `results/webcam_inference_checks/<run_name>/manifest.json`. |

## Current Model Inventory

Local checkpoint groups currently observed:

```text
ResNet-34 baselines: intentionally cleaned/reset for fair retraining
ResNet-50 baselines: intentionally cleaned/reset for fair retraining
Proposed v1 model:   intentionally cleaned/reset; not current final candidate
Proposed v2 model:   models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar observed
```

The old proposed-v1 checkpoint path, when retrained, is shared across encoder-specific comparison wrappers:

```text
models/unet_mobilenetv3_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
```

Current final-candidate proposed v2 checkpoint:

```text
models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
```

## Fair Baseline Cleanup Applied On 2026-07-10

Purpose:

```text
Reset old ResNet-34 and ResNet-50 baseline artifacts before retraining them fairly against the fixed proposed-v2 checkpoint.
```

Artifacts intentionally removed:

```text
old ResNet-34 baseline checkpoints, predictions, reports, and DBs
old ResNet-50 baseline checkpoints, predictions, reports, and DBs
old proposed-v1 checkpoint, predictions, reports, DBs, and trial checkpoint folders
results/webcam_inference_checks/
repo-level cache/temp folders
```

Artifacts intentionally preserved:

```text
.venv/
data/
exports/
results/training_logs/
models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
results/proposed_mnv3_aux_fg_e150_unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.db
models/_trial_checkpoints/proposed_mnv3_aux_fg_e150_unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1/
```

Verification commands run during cleanup finalization:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\cleanup_retrain_baselines.ps1 -Apply
.\.venv\Scripts\python.exe scripts\check_comparison_status.py --root_path . --subset test --proposed_variant v2 --write docs\COMPARISON_STATUS_V2.md
.\.venv\Scripts\python.exe scripts\check_comparison_status.py --root_path . --subset test_different_bbch --proposed_variant v2 --write docs\COMPARISON_STATUS_V2_BBCH.md
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v2 -NFolds 2 -NTrials 2 -MaxEpochs 150 -EarlyStopPatience 60 -LrSchedulerPatience 10 -RunPrefix fair_resnet50_v2 -SkipProposedTraining -PlanOnly
.\.venv\Scripts\python.exe -m py_compile train.py predict_testset.py compare_model_predictions.py evaluate_model_suite.py scripts\run_model_suite.py scripts\export_proposed_ncnn.py scripts\check_webcam_inference.py utils\manual_unet_mobilenetv3.py utils\reporting.py utils\predict.py utils\train.py
powershell -ExecutionPolicy Bypass -File .\scripts\cleanup_retrain_baselines.ps1
```

Final cleanup dry-run result:

```text
No baseline retrain cleanup targets found.
```

ResNet-50 baseline checkpoint names to recreate during fair retraining:

```text
models/fcn8s_resnet50_dil0_bilin1_pre1.pth.tar
models/fcn16s_resnet50_dil0_bilin1_pre1.pth.tar
models/fcn32s_resnet50_dil0_bilin1_pre1.pth.tar
models/unet_resnet50_dil0_bilin1_pre1.pth.tar
models/dlplus_resnet50_dil0_bilin1_pre1.pth.tar
```

These files were intentionally removed on 2026-07-10 to prepare a fair retraining batch. Do not claim they are current until they exist again and the status checker reports them as present.

## Edge NCNN Export Preparation

Target:

```text
architecture = unet_mobilenetv3_aux
encoder      = mobilenetv3_large
checkpoint   = models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
output dir   = exports/ncnn/unet_mobilenetv3_aux_256/
```

Current phase status:

| Phase | Status | Evidence |
| --- | --- | --- |
| Export script | Implemented | `scripts/export_proposed_ncnn.py` |
| Segmentation-only wrapper | Implemented | Export script returns `output["segmentation"]` logits from the proposed-v2 model. |
| Edge documentation | Implemented | `docs/EDGE_NCNN_RASPBERRY_PI5.md`, `docs/EDGE_INFERENCE_IMPLEMENTATION_NOTES.md` |
| README / AGENTS pointers | Implemented | README edge section and AGENTS export evidence rules. |
| Local dependency check | Checked | `pnnx`, `onnx`, `onnxruntime`, and Python `ncnn` were not installed in the local `.venv` during the export-prep audit. |
| TorchScript export attempt | Checked | `model.pt` written; dummy max_abs_error `1.919e-05`, sample patch max_abs_error `5.817e-05`, argmax agreement `1.0`. |
| Direct ONNX fallback | Blocked | `model.onnx` was not written because PyTorch ONNX export reported missing `onnxscript`. |
| NCNN `.param/.bin` generation | Checked | `model.ncnn.param` and `model.ncnn.bin` exist under `exports/ncnn/unet_mobilenetv3_aux_256/`; generated by pnnx CLI. |
| NCNN runtime parity | Blocked | Python `ncnn` binding was not installed locally, so NCNN argmax agreement was not measured. |
| Raspberry Pi 5 runtime validation | Not run | Later phase; no Pi-side FPS/latency claim exists. |
| OpenCV still-image checker | Checked previously; artifacts cleaned | `scripts/check_webcam_inference.py --mode pytorch --image data\test\patches\img\test_p_00000_img.png --device cpu --run_name image_check_verify` processed 1 frame in an earlier check. Its output folder was removed during the 2026-07-10 fair baseline cleanup. Rerun the command when fresh local inference evidence is needed. |
| OpenCV GUI layout | Implemented, package state checked | `--view gui` displays original frame, prediction overlay, class mask, status panel, class coverage, controls, and watermark `code by gedxxe`. The venv was initially observed with `opencv-python-headless 4.13.0.92`; after reinstall, `opencv-python 4.13.0.92` is installed and `cv2.getBuildInformation()` reports `GUI: WIN32UI`. `pip check` still reports an Albumentations metadata conflict because Albumentations declares `opencv-python-headless`; imports for `albumentations`, `albucore`, and `cv2` were checked successfully. The script preflights HighGUI and avoids secondary `destroyAllWindows()` tracebacks. |
| OpenCV webcam checker | Implemented; live evidence is manifest-based | Agent-side webcam probe previously failed because the execution environment could not open `--source 0`. The user later reported the GUI path runs after OpenCV GUI reinstall. Old webcam captures were cleaned on 2026-07-10. Treat a live run as citeable only when a fresh `results/webcam_inference_checks/<run_name>/manifest.json` exists. |
| Webcam demo command | Documented, not run here | README and edge docs now include `--view gui --max_frames 300 --save_video results\webcam_inference_checks\webcam_cuda_gui_demo.mp4 --run_name webcam_cuda_gui_demo`. Do not claim the demo exists until that run writes a manifest/video. |
| Headless webcam benchmark command | Documented, not run here | README and edge docs now include `--view headless --max_frames 300 --save_every 100 --run_name webcam_cuda_headless_300`. Treat this as a PC-side PyTorch timing check, not a Raspberry Pi 5 result. |

Required export command:

```powershell
.\.venv\Scripts\python.exe scripts\export_proposed_ncnn.py --checkpoint models\unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar --output_dir exports\ncnn\unet_mobilenetv3_aux_256 --input_size 256 --device cpu
```

Evidence required before saying NCNN export is complete:

```text
exports/ncnn/unet_mobilenetv3_aux_256/model.ncnn.param
exports/ncnn/unet_mobilenetv3_aux_256/model.ncnn.bin
exports/ncnn/unet_mobilenetv3_aux_256/export_manifest.json
exports/ncnn/unet_mobilenetv3_aux_256/parity_report.json
```

Current local export artifact status:

```text
overall_status = ncnn_exported_pending_pi_runtime_parity
onnx_exported  = false
ncnn_exported  = true
```

Do not report Raspberry Pi 5 real-time readiness yet. The remaining correctness check is NCNN runtime argmax agreement on the device or with a local Python NCNN binding.

Verification commands run for this export-prep change:

```powershell
.\.venv\Scripts\python.exe -m py_compile train.py predict_testset.py compare_model_predictions.py evaluate_model_suite.py scripts\run_model_suite.py scripts\export_proposed_ncnn.py utils\patch_utils.py utils\reporting.py utils\predict.py utils\train.py utils\manual_unet_mobilenetv3.py utils\model_outputs.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m pytest
```

Result:

```text
py_compile: passed
unittest:   passed, 20 tests
pytest:     not run because pytest is not installed in .venv
```

OpenCV inference checker command to verify without webcam:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --image data\test\patches\img\test_p_00000_img.png --device cpu --run_name image_check
```

Command actually run for still-image check:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --image data\test\patches\img\test_p_00000_img.png --device cpu --run_name image_check_verify
```

Observed output from the earlier run, now cleaned:

```text
results/webcam_inference_checks/image_check_verify/
  frame_000000.jpg
  mask_000000.png
  overlay_000000.jpg
  preview_000000.jpg
  manifest.json
```

Command actually run after the GUI preflight fix:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --image data\test\patches\img\test_p_00000_img.png --device cpu --view gui --run_name gui_preflight_expected_fail
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --image data\test\patches\img\test_p_00000_img.png --device cpu --view headless --run_name headless_after_gui_fix
```

Result:

```text
GUI mode: clean user-facing ERROR because the venv contains opencv-python-headless 4.13.0.92.
Headless mode: processed 1 frame and wrote results/webcam_inference_checks/headless_after_gui_fix/manifest.json. That output was later cleaned during the 2026-07-10 fair baseline reset.
```

Webcam command for local machine with camera access:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --source 0 --device cuda --max_frames 100
```

GUI command for local machine with camera access:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --source 0 --device cuda --view gui
```

Use `--view gui` only when OpenCV GUI support exists. For automated checks or old venvs that still use `opencv-python-headless`, use `--view headless`.

Webcam probe actually attempted in this environment:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --source 0 --device cpu --max_frames 1 --no_save_last --run_name webcam_source0_probe --output_dir tmp\webcam_probe
```

Result:

```text
blocked: Could not open OpenCV source '0'
```

## Encoder-Specific Automation Status

| Encoder | Script | PlanOnly validated | Default batch size | Combined report path |
| --- | --- | --- | ---: | --- |
| `resnet18` | `scripts/run_resnet18_architecture_comparison.ps1` | Yes | 8 | `results/reports/test/architecture_comparison_resnet18/` full, `baseline_comparison_resnet18/` baseline-only |
| `resnet34` | `scripts/run_resnet34_architecture_comparison.ps1` | Yes | 8 | `results/reports/test/architecture_comparison_resnet34/` full, `baseline_comparison_resnet34/` baseline-only |
| `resnet50` | `scripts/run_resnet50_architecture_comparison.ps1` | Yes | 4 | `results/reports/test/architecture_comparison_resnet50/` full, `baseline_comparison_resnet50/` baseline-only |
| `resnet101` | `scripts/run_resnet101_architecture_comparison.ps1` | Yes | 2 | `results/reports/test/architecture_comparison_resnet101/` full, `baseline_comparison_resnet101/` baseline-only |
| proposed | `scripts/run_proposed_model.ps1` | Yes | 8 | `results/reports/test/proposed_model/` via Python suite |

Complete evaluation reports use these paths:

```text
results/reports/test/full_evaluation_<encoder>/
results/reports/test/baseline_evaluation_<encoder>/
results/reports/test/proposed_evaluation/
results/reports/test/ablation_evaluation_<encoder>/
```

## Historical Six-Model ResNet-34 Comparison Target

Earlier progress-grade comparison set, now reset for fair retraining:

```text
fcn8s_resnet34
fcn16s_resnet34
fcn32s_resnet34
unet_resnet34
dlplus_resnet34
unet_mobilenetv3
```

| Model | Checkpoint status | Prediction status | Report status | Notes |
| --- | --- | --- | --- | --- |
| `fcn8s_resnet34` | Cleaned | Cleaned | Cleaned | Removed on 2026-07-10 for fair retraining. |
| `fcn16s_resnet34` | Cleaned | Cleaned | Cleaned | Removed on 2026-07-10 for fair retraining. |
| `fcn32s_resnet34` | Cleaned | Cleaned | Cleaned | Removed on 2026-07-10 for fair retraining. |
| `unet_resnet34` | Cleaned | Cleaned | Cleaned | Removed on 2026-07-10 for fair retraining. |
| `dlplus_resnet34` | Cleaned | Cleaned | Cleaned | Removed on 2026-07-10 for fair retraining. |
| `unet_mobilenetv3` | Cleaned | Cleaned | Cleaned | Old proposed-v1 artifact removed because the current candidate is proposed v2. |

The five baseline ResNet-34 model-level reports and the v1 proposed model-level report are no longer present locally. This section remains only to document the old comparison identity.

Required combined report:

```text
results/reports/test/architecture_comparison_resnet34/
  metrics_summary.csv
  metrics_per_image.csv
  qualitative_grid.png
```

## Proposed History and Historical V2 ResNet-50 Results

Important status:

```text
Reports generated before 2026-07-09 used stale sigmoid-threshold prediction post-processing.
The prediction code now uses argmax over raw logits. Regenerate predictions/reports before citing final confusion matrices.
```

Historical report paths:

```text
results/reports/test/unet_mobilenetv3/
results/reports/test/architecture_comparison_resnet50/
```

Direct patch-level audit after the argmax fix:

```text
current canonical proposed v1, argmax logits:
mIoU      = 0.813129
meanDice  = 0.890594
pixelAcc  = 0.994128
Sorghum F1 / Recall = 0.8797 / 0.9419
Weed F1 / Recall    = 0.7946 / 0.8070
```

This was a direct patch-level audit, not the current final-candidate report.

Proposed v2 auxiliary-foreground run used for the current final candidate:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -Variant v2 -NFolds 2 -NTrials 2 -MaxEpochs 150 -BatchSize 8 -NumWorkers 2 -RunPrefix proposed_mnv3_aux_fg_e150 -ClassWeightMax 8 -ClassWeightStrategy inverse_frequency -CeWeight 1.0 -DiceWeight 1.0 -ForegroundAuxWeight 0.3 -ValidationLoss foreground_macro_f1 -EarlyStopPatience 60 -LrSchedulerPatience 10
```

Current proposed v2 status:

```text
Optuna DB: results/proposed_mnv3_aux_fg_e150_unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.db
Trial 0: COMPLETE, value 0.12375178933143616, final_epoch [149, 149]
Trial 1: COMPLETE, value 0.13236075639724731, final_epoch [149, 149]
Best trial: Trial 0
Trial checkpoints:
  models/_trial_checkpoints/proposed_mnv3_aux_fg_e150_unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1/trial_000_unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
  models/_trial_checkpoints/proposed_mnv3_aux_fg_e150_unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1/trial_001_unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
Canonical checkpoint: models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
Canonical checkpoint matches trial 0 by SHA256 in the local audit.
```

Historical ResNet-50 full-v2 report on `test`, cleaned on 2026-07-10:

```text
results/reports/test/full_v2_evaluation_resnet50/evaluation_summary.csv
results/reports/test/full_v2_evaluation_resnet50/metrics_summary.csv
results/reports/test/architecture_comparison_resnet50_proposed_v2/metrics_summary.csv
```

Historical `test` summary after regeneration:

```text
dlplus_resnet50          mIoU=0.822747  meanDice=0.897055  model=101.99 MiB  GFLOPs=897.04
unet_resnet50           mIoU=0.822374  meanDice=0.896798  model=124.27 MiB  GFLOPs=225.03
unet_mobilenetv3_aux    mIoU=0.816617  meanDice=0.892801  model=15.28 MiB   GFLOPs=17.48
```

Command to retrain the ResNet-50 baselines against the fixed proposed-v2 checkpoint after cleanup:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v2 -SkipProposedTraining -NFolds 2 -NTrials 2 -MaxEpochs 150 -BatchSize 4 -NumWorkers 2 -RunPrefix fair_resnet50_v2 -EarlyStopPatience 60 -LrSchedulerPatience 10
```

Command to evaluate the same completed checkpoints on `test_different_bbch` after the BBCH pairing fix:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v2 -SkipTraining -Subset test_different_bbch -BatchSize 4 -NumWorkers 0 -RunPrefix fair_resnet50_v2_bbch
```

Historical `test_different_bbch` report after the image-mask pairing fix, cleaned on 2026-07-10:

```text
results/reports/test_different_bbch/full_v2_evaluation_resnet50/evaluation_summary.csv
results/reports/test_different_bbch/architecture_comparison_resnet50_proposed_v2/metrics_summary.csv

unet_mobilenetv3_aux    mIoU=0.734414  meanDice=0.836219  pixelAcc=0.958934
unet_resnet50           mIoU=0.726654  meanDice=0.830046  pixelAcc=0.961729
dlplus_resnet50         mIoU=0.670400  meanDice=0.784167  pixelAcc=0.953486
```

Old v1 restore path if proposed v2 needs to be revisited. The v1 checkpoint was cleaned on 2026-07-10, so this would retrain v1 rather than restore an existing artifact:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v1 -SkipProposedTraining -NFolds 2 -NTrials 2 -MaxEpochs 150 -BatchSize 4 -NumWorkers 2 -RunPrefix restore_v1_resnet50_argmax
```

## Historical ResNet-34 Progress Run

This remains documented for reproducing the earlier progress-grade comparison. It is not the current final-candidate workflow.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -BatchSize 8 -NumWorkers 2 -RunPrefix archcmp_resnet34
```

What this run means:

```text
Six-model comparison: five fixed-ResNet-34 baselines plus proposed MobileNetV3 U-Net, 2-fold CV, 1 Optuna trial, max 10 epochs, batch size 8, CUDA AMP enabled.
```

This is progress-grade, not final-grade.

Other encoder progress commands:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet18_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -BatchSize 8 -NumWorkers 2 -RunPrefix archcmp_resnet18
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -BatchSize 4 -NumWorkers 2 -RunPrefix archcmp_resnet50
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet101_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 10 -BatchSize 2 -NumWorkers 2 -RunPrefix archcmp_resnet101
```

Parameter interpretation:

| Parameter | Meaning | Current recommendation |
| --- | --- | --- |
| `NFolds` | Number of cross-validation splits. More folds are slower but more stable. | `2` for progress, `4` for final-style comparison. |
| `MaxEpochs` | Maximum epochs per fold. Early stopping may end earlier. | `10` quick, `20` stronger progress, `100` final-style. |
| `EarlyStopPatience` | Epochs without validation improvement before stopping a fold. | Increase with long runs; `25` is a reasonable start for `MaxEpochs` around `120`. |
| `LrSchedulerPatience` | Epoch patience before `ReduceLROnPlateau` lowers LR. | Keep `5` normally; use `8-10` only for long slow runs. |
| `NTrials` | Optuna trial budget. | `1` for fair fixed-budget architecture comparison. |
| `BatchSize` | Patch count per training batch; primary VRAM lever. | `8` for ResNet-18/34, `4` for ResNet-50, `2` for ResNet-101. |
| `NumWorkers` | Training DataLoader worker count. | `2` normally, `0` if Windows multiprocessing fails. |
| `Device` | Runtime device for training and prediction. | `cuda` for the RTX 5060 run, `auto` for portable checks, `cpu` only for quick validation/debug. |
| `RunPrefix` | Experiment namespace for result DBs. | Use unique prefixes for new experiment batches. |
| `CleanStudy` | Deletes existing trials for the same study before training. | Use only for intentional clean reruns. |

For materially different proposed-model tuning, use a new `RunPrefix` rather than appending to an old Optuna study. Use `CleanStudy` only when the old study should intentionally be replaced.

## Historical ResNet-34 Final-Style Run

This remains documented for reproducing the earlier v1-style final setting. The current final-candidate path is the proposed v2 + ResNet-50 comparison shown in the section above.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 4 -MaxEpochs 100 -BatchSize 8 -NumWorkers 2 -RunPrefix final_archcmp_resnet34
```

What this run means:

```text
Six-model comparison: five fixed-ResNet-34 baselines plus proposed MobileNetV3 U-Net, 4-fold CV, 1 Optuna trial, max 100 epochs, batch size 8, CUDA AMP enabled.
```

This is the cleaner setting for final reporting, assuming all six model targets complete.

## Latest Verified Commands

The following validation commands are safe to rerun without starting training:

```powershell
.\.venv\Scripts\python.exe -m py_compile train.py small_training.py predict_testset.py compare_model_predictions.py evaluate_model_suite.py utils\losses.py utils\parser.py utils\train.py utils\predict.py utils\model_outputs.py utils\manual_unet_mobilenetv3.py utils\manual_resnet.py scripts\check_comparison_status.py scripts\run_model_suite.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -Variant v2 -NFolds 2 -NTrials 2 -MaxEpochs 150 -EarlyStopPatience 60 -LrSchedulerPatience 10 -RunPrefix proposed_mnv3_aux_fg_e150 -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v2 -NFolds 2 -NTrials 2 -MaxEpochs 150 -EarlyStopPatience 60 -LrSchedulerPatience 10 -RunPrefix fair_resnet50_v2 -SkipProposedTraining -PlanOnly
.\.venv\Scripts\python.exe evaluate_model_suite.py proposed_v2 --root_path . --subset test --device cuda --warmup_iterations 0 --benchmark_iterations 1 --report_dir results\reports\test\manual_verification_proposed_v2_eff --max_examples 1
.\.venv\Scripts\python.exe scripts\check_dataset.py --require-patches
.\.venv\Scripts\python.exe small_training.py unet_mobilenetv3 mobilenetv3_large --root_path . --device cpu --batch_size 2 --num_workers 0 --n_folds 2 --max_epochs 0 --loss ce_dice --class_weights auto --no-pretrained
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet18_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 1 -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 1 -BaselineOnly -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -NFolds 2 -MaxEpochs 1 -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 1 -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 1 -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet101_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 1 -PlanOnly
.\.venv\Scripts\python.exe scripts\run_model_suite.py baseline --encoder resnet34 --n_folds 2 --max_epochs 1 --plan_only
.\.venv\Scripts\python.exe scripts\run_model_suite.py proposed --n_folds 2 --max_epochs 1 --plan_only
.\.venv\Scripts\python.exe scripts\run_model_suite.py full --encoder resnet34 --n_folds 2 --max_epochs 1 --plan_only
.\.venv\Scripts\python.exe scripts\run_model_suite.py full --encoder resnet50 --skip_training --plan_only
.\.venv\Scripts\python.exe scripts\run_model_suite.py ablation --encoder resnet34 --n_folds 2 --max_epochs 1 --skip_efficiency --plan_only
.\.venv\Scripts\python.exe scripts\check_comparison_status.py --root_path . --subset test --write docs\COMPARISON_STATUS.md
```

## Latest Final-Candidate Audit

Date: 2026-07-10

Audit result:

```text
Proposed v2 auxiliary foreground path is implemented, trained, checkpoint-published, and evaluated against the ResNet-50 baseline set on test and test_different_bbch.
```

Checks completed:

```text
Python compile: passed
Unit tests: 20 passed
Full ResNet-34 Python suite PlanOnly: passed
ResNet-18/34/50/101 PowerShell wrappers PlanOnly: passed
Proposed evaluator with minimal CUDA efficiency path: passed
Status checker refreshed docs/COMPARISON_STATUS.md
Full ResNet-50 report-only evaluator path: passed
ResNet-18/34/50/101 PowerShell full-mode prediction/evaluation PlanOnly after core delegation: passed
ResNet-18/50/101 PowerShell full-mode training PlanOnly after core delegation: passed
Proposed v2 PowerShell PlanOnly with NFolds=2, NTrials=2, MaxEpochs=150, EarlyStopPatience=60: passed
Full v2 ResNet-50 PowerShell PlanOnly with SkipTraining: passed
Status checker proposed_variant=v2: passed
Proposed v2 CUDA forward/backward check with auxiliary loss: passed
Loss factory check for dice, ce, ce_dice, and ce_dice_aux_foreground: passed
Proposed v2 canonical checkpoint: present and protected as the current final-candidate checkpoint
Proposed v2 active-run audit: complete; trial 0 was published as the canonical checkpoint
test full_v2 ResNet-50 evaluation: passed
test_different_bbch full_v2 ResNet-50 evaluation: passed
Dataset check with required patches: passed
Conservative cleanup dry-run/apply path: passed
```

Operational note:

```text
Efficiency benchmarking is fail-soft. If Params/GFLOPs/FPS fails for one model, the evaluator records the error in efficiency_error and still writes segmentation metrics and summary tables.
```

Current generated result state should be checked in:

```text
docs/COMPARISON_STATUS.md
docs/COMPARISON_STATUS_V2.md
```

## Open Gaps

| Gap | Impact | Next action |
| --- | --- | --- |
| ResNet-18/101 architecture comparisons are not trained yet | Cannot claim encoder-specific comparisons for those encoders | Run the corresponding encoder script only when needed. |
| Reports generated before the argmax post-processing fix are stale | Old confusion matrices may understate/shift class performance | Cite only regenerated reports from 2026-07-09 or newer. |
| Final paper claims still need careful wording | `test` has 3 raw images and `test_different_bbch` has 2 raw images, even though pixel counts are large | Frame results as local dataset evidence and avoid broad generalization claims. |
| Urgent run is not final-grade if using 2 folds and 10 epochs | Results are useful for progress, not final thesis claims | Rerun with 4 folds and 100 epochs for final comparison. |
| Results are pixel-level metrics | Does not include object-level agronomic metrics | Add object-level analysis only if required by supervisor scope. |
