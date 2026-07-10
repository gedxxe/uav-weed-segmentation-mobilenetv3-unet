# Windows RTX 5060 Training Runbook

This runbook is the exact local path for training this repository on Windows with Python 3.13.12 and an RTX 5060 8 GB GPU.

## 1. Confirm Python Launcher

List installed Python launch targets:

```powershell
py -0p
```

The required target is:

```powershell
py -3.13 --version
```

Expected:

```text
Python 3.13.12
```

If plain `py --version` prints another version, keep using `py -3.13` in this repository.

## 2. Create and Install Environment

Recommended:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_cuda.ps1
```

Manual equivalent:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip "setuptools<82" wheel
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\verify_cuda.py
```

The PyTorch step must happen before `requirements.txt`. Otherwise pip can satisfy `kornia` by installing a CPU-only PyTorch wheel.

## 3. Verify CUDA

Run:

```powershell
.\.venv\Scripts\python.exe scripts\verify_cuda.py
```

Pass condition:

- `CUDA available: True`
- GPU name is your RTX 5060
- VRAM is close to 8 GiB
- CUDA tensor quick check prints a numeric result

If CUDA is false, check NVIDIA driver first, then reinstall PyTorch using the CUDA wheel command above.

## 4. Verify Raw Dataset

Run:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py
```

Expected raw layout:

```text
data/trainval/img/*.jpg
data/trainval/msk/*.png
data/test/img/*.jpg
data/test/msk/*.png
data/test_different_bbch/img/*.jpg
data/test_different_bbch/msk/*.png
```

The raw image and mask counts must match for every subset.

## 5. Generate Patches

Run once after installing dependencies:

```powershell
.\.venv\Scripts\python.exe save_patches.py --root_path .
```

Then verify patch folders:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py --require-patches
```

Training reads from `data/trainval/patches/img` and `data/trainval/patches/msk`, not directly from the raw `.jpg` files.

## 6. Quick Validation Training

Run every command in this section separately. Do not paste the quick-validation command and the full-training command onto one physical PowerShell line.

Run one epoch on one fold:

```powershell
.\.venv\Scripts\python.exe small_training.py unet resnet34 --root_path . --device cuda --batch_size 8 --num_workers 2 --max_epochs 1
```

Use this before a long Optuna run. It verifies:

- model construction
- pretrained ResNet download/cache path if `--pretrained` is enabled
- DataLoader worker behavior
- CUDA mixed precision
- loss calculation and validation

If the quick validation run fails with CUDA out-of-memory:

```powershell
.\.venv\Scripts\python.exe small_training.py unet resnet34 --root_path . --device cuda --batch_size 4 --num_workers 2 --max_epochs 1
```

If it fails inside Windows multiprocessing/DataLoader:

```powershell
.\.venv\Scripts\python.exe small_training.py unet resnet34 --root_path . --device cuda --batch_size 8 --num_workers 0 --max_epochs 1
```

Pretrained ResNet weights are cached under:

```text
.cache/torch/hub/checkpoints/
```

This avoids relying on write permission to `C:\Users\<user>\.cache\torch`.

## 7. Full Training Baseline

Run this only after the quick-validation command has completed and the `PS ...>` prompt has returned.

Start with one Optuna trial:

```powershell
.\.venv\Scripts\python.exe train.py unet resnet34 --root_path . --device cuda --batch_size 8 --num_workers 2 --n_trials 1 --max_epochs 100 --run_prefix rtx5060
```

Outputs:

```text
results/rtx5060_unet_resnet34_dil0_bilin1_pre1.db
```

The command above saves Optuna study results, not a prediction checkpoint. Add `--save_checkpoint` when you want a local model file for `predict_testset.py`:

```powershell
.\.venv\Scripts\python.exe train.py unet resnet34 --root_path . --device cuda --batch_size 8 --num_workers 2 --n_trials 1 --max_epochs 100 --run_prefix rtx5060 --save_checkpoint
```

Expected checkpoint path:

```text
models/unet_resnet34_dil0_bilin1_pre1.pth.tar
```

Use it for prediction:

```powershell
.\.venv\Scripts\python.exe predict_testset.py models\unet_resnet34_dil0_bilin1_pre1.pth.tar test --root_path . --device cuda --num_workers 0 --model_name rtx5060_unet_resnet34 --output_dir results\predictions\test\rtx5060_unet_resnet34
```

Prediction evaluates one checkpoint at a time. The command above uses:

- raw input images from `data/test/img/*.jpg`
- ground-truth masks from `data/test/msk/*.png`
- generated prediction masks in `results/predictions/test/rtx5060_unet_resnet34`
- generated report files in `results/reports/test/rtx5060_unet_resnet34`

Report outputs:

```text
results/reports/test/rtx5060_unet_resnet34/
  README.md
  manifest.json
  metrics_summary.csv
  metrics_summary.json
  metrics_per_image.csv
  confusion_matrix_rtx5060_unet_resnet34.csv
  confusion_matrix_rtx5060_unet_resnet34.png
  qualitative_grid.png
```

`qualitative_grid.png` contains representative crops with raw image, ground truth, prediction, prediction overlay, and error-map rows. Error colors are black for correct pixels, red for false-positive foreground, blue for false-negative foreground, and yellow for wrong foreground class.

## 8. Compare Several Models

For the full six-model comparison procedure across fixed ResNet baselines plus `unet_mobilenetv3`, use `docs/MODEL_COMPARISON_RUNBOOK.md`. Current status and gaps are tracked in `docs/DEVELOPMENT_TRACKER.md`.

With the current local checkpoint, this command builds the one-model prediction report:

```powershell
.\.venv\Scripts\python.exe predict_testset.py models\unet_resnet34_dil0_bilin1_pre1.pth.tar test --root_path . --device cuda --num_workers 0 --model_name unet_resnet34 --output_dir results\predictions\test\unet_resnet34
```

For a real multi-model comparison, run prediction once per checkpoint and keep each model in a separate prediction directory. After a second folder such as `results\predictions\test\other_model` exists, build the comparison report:

```powershell
.\.venv\Scripts\python.exe compare_model_predictions.py test --root_path . --prediction unet_resnet34=results\predictions\test\unet_resnet34 --prediction other_model=results\predictions\test\other_model --report_dir results\reports\test\comparison
```

This report compares every listed model against the same raw images and ground-truth masks. It does not retrain or rerun inference; it only evaluates saved prediction masks.

## 9. Useful Training Variants

Disable pretrained ResNet weights:

```powershell
.\.venv\Scripts\python.exe train.py unet resnet34 --root_path . --device cuda --no-pretrained --batch_size 8 --n_trials 1
```

Run CPU-only for debugging:

```powershell
.\.venv\Scripts\python.exe train.py unet resnet34 --root_path . --device cpu --batch_size 2 --num_workers 0 --n_trials 1 --max_epochs 1
```

Try a lighter encoder:

```powershell
.\.venv\Scripts\python.exe train.py unet resnet18 --root_path . --device cuda --batch_size 8 --n_trials 1
```

Run a longer search:

```powershell
.\.venv\Scripts\python.exe train.py unet resnet34 --root_path . --device cuda --batch_size 8 --num_workers 2 --n_trials 10 --max_epochs 100 --run_prefix rtx5060_search
```

## 10. Failure Diagnosis

`No patch images/masks found`

Run:

```powershell
.\.venv\Scripts\python.exe save_patches.py --root_path .
```

`Model file not found`

Check whether a model exists:

```powershell
Get-ChildItem .\models\*.pt, .\models\*.pth, .\models\*.pth.tar -ErrorAction SilentlyContinue
```

If no file is listed, prediction cannot run yet. Download/copy the trained model into `models/`, or rerun training with `--save_checkpoint`.

`CUDA was requested, but torch.cuda.is_available() is False`

Run:

```powershell
.\.venv\Scripts\python.exe scripts\verify_cuda.py
```

Then reinstall the CUDA wheel if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

`CUDA out of memory`

Use one or more:

```powershell
--batch_size 4
--batch_size 2
```

Start by reducing batch size. Disable AMP only when debugging numerical issues.

DataLoader hangs or worker crash on Windows:

```powershell
--num_workers 0
```

If prediction prints repeated `Using Seed 42` / `Predicting...` lines or raises `An attempt has been made to start a new process before the current process has finished its bootstrapping phase`, rerun prediction with:

```powershell
--num_workers 0
```

Pretrained weight download fails:

Use:

```powershell
--no-pretrained
```

or rerun when internet access is available. The pretrained ResNet weights are fetched by PyTorch when first used.

## 11. What Changed for Local Training

- Training device is now configurable with `--device auto|cuda|cpu`.
- CUDA AMP can be toggled with `--amp` / `--no-amp`.
- Batch size default is `8` instead of `100`.
- Optuna trial default is `1` instead of `50`.
- DataLoader workers and pinned memory are configurable.
- Validation no longer keeps the full validation tensor on GPU for Dice-only validation; predictions are moved to CPU before the full validation Dice loss is computed. Proposed-model runs can use `--validation_loss macro_f1`, `--validation_loss weed_f1`, or `--validation_loss foreground_macro_f1` to select checkpoints with class-balanced validation metrics.
- UNet and DeepLabV3+ now respect `--pretrained` / `--no-pretrained`.
- Dataset and patch checks now raise explicit path-level errors.
- Patch generation no longer depends on `patchify`, because that package requires `numpy<2` and blocks the Python 3.13 runtime.
- Prediction now writes a model-labeled report with metrics, confusion matrices, raw/ground-truth/prediction grids, and optional multi-model comparison.
