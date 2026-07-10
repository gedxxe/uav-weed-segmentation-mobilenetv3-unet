# Proposed Model Notes: U-Net MobileNetV3

Status: proposed v1 and proposed v2 are implemented as local trainable architectures. Do not claim either variant is better than a baseline unless the local checkpoint and refreshed reports exist.

Current final-candidate for ResNet-50 comparison work:

```text
architecture = unet_mobilenetv3_aux
encoder      = mobilenetv3_large
checkpoint   = models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
```

The old ResNet-34/ResNet-50 baseline artifacts were intentionally cleaned on 2026-07-10 for fair retraining. Treat old comparison numbers as historical notes until the corresponding checkpoints and reports are regenerated.

Local source studied:

```text
C:\Users\gedee\Downloads\Improved UNet Lightweight Network for Semantic Segmentation ofWeed.pdf
```

Paper:

```text
Yu Zuo and Wenwen Li, "An Improved UNet Lightweight Network for Semantic Segmentation of Weed Images in Corn Fields", CMC, 2024, vol. 79, no. 3.
DOI: 10.32604/cmc.2024.049805
Published: 2024-06-20
```

## Research Position In This Repo

The intended research framing is:

```text
Proposed model:
  improved U-Net with MobileNetV3_Large encoder, PPM skip/context module, and SE/H-swish decoder

Baseline models:
  fcn8s, fcn16s, fcn32s, unet, dlplus
  with resnet18, resnet34, resnet50, resnet101 encoders
```

Existing baseline matrix:

```text
5 architectures x 4 ResNet encoders = 20 baseline model combinations
```

Do not merge the proposed model into the existing ResNet encoder matrix conceptually. The proposed model should be treated as a separate architecture family, then compared against the selected baselines under the same dataset split, seed, fold count, epoch budget, batch policy, and evaluation report pipeline.

## Paper Architecture Summary

The paper is not only "U-Net with MobileNetV3". The full proposed network combines three main changes:

| Component | Paper choice | Purpose |
| --- | --- | --- |
| Encoder/compression path | MobileNetV3_Large, pretrained on ImageNet-1K | Reduce parameters and improve feature extraction on small agricultural datasets. |
| Skip/context module | Pyramid Pooling Module (PPM) on skip connections | Capture multi-scale context and reduce the semantic gap between shallow and deep features. |
| Decoder/extension path | Modified decoder using H-swish and SE attention | Recover segmentation detail while dynamically reweighting channels. |

The paper names intermediate ablations:

| Name in paper | Meaning |
| --- | --- |
| `Unet` | Original U-Net baseline. |
| `MN_unet` | U-Net using MobileNetV3_Large as backbone. |
| `MN_PPM_unet` | MobileNetV3 U-Net plus PPM. |
| `MN_PPM_MD_unet` / `ours` | MobileNetV3 U-Net plus PPM plus modified decoder. |

The proposed model target for this repo should most likely map to the final paper model:

```text
mnv3_ppm_md_unet
```

A shorter CLI name can be used later, for example:

```text
unet_mobilenetv3
```

but the documentation should state clearly that it means MobileNetV3_Large + PPM + modified decoder, not the paper's `MN_unet` ablation.

## Local Implementation

Current local command identities:

```text
proposed v1 architecture = unet_mobilenetv3
proposed v2 architecture = unet_mobilenetv3_aux
encoder                   = mobilenetv3_large
```

Implemented ablation architecture identities:

```text
unet_mobilenetv3_base  = MobileNetV3_Large U-Net without PPM and without SE decoder
unet_mobilenetv3_ppm   = MobileNetV3_Large U-Net with PPM and without SE decoder
unet_mobilenetv3       = MobileNetV3_Large U-Net with PPM and SE/H-swish decoder
unet_mobilenetv3_aux   = proposed v2; MobileNetV3_Large U-Net with PPM, SE/H-swish decoder, and an auxiliary BG-vs-vegetation foreground head
```

## Proposed V2: Auxiliary Foreground Supervision

The v2 model keeps the same encoder, PPM skip modules, and SE/H-swish decoder as v1. It adds one auxiliary head at the final decoder feature map:

```text
decoder feature
  -> segmentation head: 3 logits, Background / Sorghum / Weed
  -> foreground head:   2 logits, Background / Vegetation
```

The foreground target is derived from the same mask without extra labels:

```text
foreground_target = target > 0
```

This is an architecture modification and must be reported as proposed v2 or as an ablation row. It is not the same model as `unet_mobilenetv3`.

The training loss for v2 is:

```text
CE_Dice_3class + foreground_aux_weight * CE_foreground
```

Default local value:

```text
foreground_aux_weight = 0.3
```

The v2 checkpoint path is independent from v1:

```text
models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
```

For inference/export, v2 still exposes the 3-class semantic segmentation logits as the final output:

```text
segmentation logits: [1, 3, H, W]
class mask: argmax(segmentation_logits, channel_dimension)
```

The auxiliary foreground head is not exported as the primary edge-runtime output. It is training supervision, and its effect is already encoded in the shared encoder and decoder weights. Edge-export implementation details are kept in:

```text
docs/EDGE_NCNN_RASPBERRY_PI5.md
docs/EDGE_INFERENCE_IMPLEMENTATION_NOTES.md
```

Main implementation files:

| File | Purpose |
| --- | --- |
| `utils/manual_unet_mobilenetv3.py` | MobileNetV3_Large encoder, PPM skip modules, SE/H-swish decoder, segmentation head. |
| `utils/train.py` | Registers `unet_mobilenetv3` and `unet_mobilenetv3_aux` in `set_model()`. |
| `utils/parser.py` | Allows `unet_mobilenetv3 mobilenetv3_large` while keeping ResNet encoders for baselines. |
| `predict_testset.py` | Parses proposed checkpoint names with underscores. |
| `evaluate_model_suite.py` | Complete segmentation, efficiency, qualitative, and ablation evaluation. |
| `scripts/run_architecture_comparison_core.ps1` | Adds proposed model to the default six-model comparison. |
| `scripts/run_proposed_model.ps1` | PowerShell proposed-only wrapper. |
| `scripts/run_model_suite.py` | Python runner for `baseline`, `proposed`, `full`, and `ablation` modes. |
| `tests/` | CPU quick checks for shape and checkpoint-name parsing. |
| `.github/workflows/ci.yml` | CI quick-check workflow for compile, unit tests, and PowerShell PlanOnly validation. |

Local architecture detail:

```text
input
  -> high-resolution 3x3 stem skip at full input resolution
  -> MobileNetV3_Large features at 1/2, 1/4, 1/8, 1/16, and 1/32 resolution
  -> PPM-enhanced skips
  -> SE/H-swish decoder back to full input resolution
```

The high-resolution stem skip is intentional. It keeps a full-resolution boundary/detail path, matching the paper diagram more closely than using MobileNetV3's stride-2 stem as the first skip.

Local trainable parameter count from the implemented PyTorch model:

```text
3,980,467 parameters
```

This is close to the paper's reported 3.79M parameter target. A CUDA forward/backward quick check with batch size 8 and 256x256 patches produced output shape `(8, 3, 256, 256)` with about 1390 MiB peak allocation. Params, GFLOPs, FPS, latency, and peak GPU memory are now measured by `evaluate_model_suite.py` with a 480x480 RGB input.

Legacy v1 quick validation command:

```powershell
.\.venv\Scripts\python.exe small_training.py unet_mobilenetv3 mobilenetv3_large --root_path . --device cuda --batch_size 8 --num_workers 2 --n_folds 2 --max_epochs 1
```

Legacy v1 progress training command:

```powershell
.\.venv\Scripts\python.exe train.py unet_mobilenetv3 mobilenetv3_large --root_path . --device cuda --batch_size 8 --num_workers 2 --n_folds 2 --n_trials 1 --max_epochs 10 --run_prefix proposed_mnv3 --save_checkpoint
```

Legacy v1 proposed-only suite commands:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -NFolds 2 -MaxEpochs 10 -BatchSize 8 -NumWorkers 2 -RunPrefix proposed_mnv3
.\.venv\Scripts\python.exe scripts\run_model_suite.py proposed --n_folds 2 --max_epochs 10 --batch_size 8 --num_workers 2 --run_prefix proposed_mnv3
```

Proposed v2 command used for the current final-candidate run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -Variant v2 -NFolds 2 -NTrials 2 -MaxEpochs 150 -BatchSize 8 -NumWorkers 2 -RunPrefix proposed_mnv3_aux_fg_e150 -ClassWeightMax 8 -ClassWeightStrategy inverse_frequency -CeWeight 1.0 -DiceWeight 1.0 -ForegroundAuxWeight 0.3 -ValidationLoss foreground_macro_f1 -EarlyStopPatience 60 -LrSchedulerPatience 10
```

After v2 training, retrain the ResNet-50 baselines against the fixed proposed-v2 checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v2 -SkipProposedTraining -NFolds 2 -NTrials 2 -MaxEpochs 150 -BatchSize 4 -NumWorkers 2 -RunPrefix fair_resnet50_v2 -EarlyStopPatience 60 -LrSchedulerPatience 10
```

This comparison writes v2-specific reports:

```text
results/reports/test/architecture_comparison_resnet50_proposed_v2/
results/reports/test/full_v2_evaluation_resnet50/
```

If the five ResNet-50 baseline checkpoints already exist and only reports need regeneration, use `-SkipTraining`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v2 -SkipTraining -BatchSize 4 -NumWorkers 2 -RunPrefix fair_resnet50_v2_report
```

Evaluate the same completed checkpoints on the BBCH dataset-shift subset without retraining:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet50_architecture_comparison.ps1 -ProposedVariant v2 -SkipTraining -Subset test_different_bbch -BatchSize 4 -NumWorkers 0 -RunPrefix fair_resnet50_v2_bbch
```

Ablation suite command:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_suite.py ablation --encoder resnet34 --n_folds 2 --max_epochs 20 --batch_size 8 --num_workers 2 --run_prefix ablation_mnv3_resnet34
```

Evaluate existing ablation predictions/checkpoints without retraining:

```powershell
.\.venv\Scripts\python.exe evaluate_model_suite.py ablation --encoder resnet34 --root_path . --subset test --device cuda
```

Current recommended proposed-model loss recipe:

```text
loss: ce_dice
class_weights: auto
class_weight_max: 5.0
class_weight_strategy: inverse_frequency
ce_weight: 1.0
dice_weight: 1.0
validation_loss: macro_f1
```

This is intentional. The first observed proposed-model failure was not missing plant shapes; it was class confusion where many true `Weed` pixels were predicted as `Sorghum`. A combined CE + Dice objective is more appropriate for that failure mode than Dice-only training. `validation_loss=macro_f1` selects the checkpoint by validation macro F1, which is closer to the report objective than Dice-only checkpoint selection.

Historical v1 follow-up before the auxiliary-foreground v2 run:

```text
The proposed model is lightweight and strong on Weed, but still trails unet_resnet50 mainly on Sorghum recall/F1.
The historical v1 tuning target was balanced foreground performance, not architecture widening.
```

This note belongs to the v1 tuning path. The current final-candidate path is `unet_mobilenetv3_aux`; use the v2 commands above unless intentionally reproducing the older v1 tuning experiment.

Historical v1 Sorghum/Weed balanced foreground progress command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -NFolds 2 -MaxEpochs 40 -BatchSize 8 -NumWorkers 2 -RunPrefix proposed_mnv3_fgmacro_sqrt -ClassWeightMax 8 -ClassWeightStrategy sqrt_inverse -CeWeight 1.2 -DiceWeight 1.0 -ValidationLoss foreground_macro_f1
```

Historical longer v1 run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposed_model.ps1 -NFolds 4 -NTrials 3 -MaxEpochs 150 -BatchSize 8 -NumWorkers 2 -RunPrefix proposed_mnv3_fgmacro_sqrt_e150 -ClassWeightMax 8 -ClassWeightStrategy sqrt_inverse -CeWeight 1.2 -DiceWeight 1.0 -ValidationLoss foreground_macro_f1 -EarlyStopPatience 60 -LrSchedulerPatience 10
```

Use a fresh `RunPrefix` when changing the recipe. When `NTrials > 1`, training saves per-trial checkpoints and publishes the best available trial checkpoint to the canonical proposed checkpoint path after Optuna finishes.

Training control note:

```text
Early stopping is consecutive after 2026-07-08: every validation improvement resets the patience counter.
```

Historical v1 prediction command after its checkpoint exists:

```powershell
.\.venv\Scripts\python.exe predict_testset.py models\unet_mobilenetv3_mobilenetv3_large_dil0_bilin1_pre1.pth.tar test --root_path . --device cuda --num_workers 0 --model_name unet_mobilenetv3 --output_dir results\predictions\test\unet_mobilenetv3
```

Current v2 prediction command when only regenerating the proposed-v2 prediction folder:

```powershell
.\.venv\Scripts\python.exe predict_testset.py models\unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar test --root_path . --device cuda --num_workers 0 --model_name unet_mobilenetv3_aux --output_dir results\predictions\test\unet_mobilenetv3_aux
```

Checkpoint overwrite note:

```text
The proposed checkpoint name is independent of the baseline encoder wrapper.
Running several encoder comparison scripts can retrain and overwrite:
models/unet_mobilenetv3_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
```

Proposed v2 does not overwrite that v1 checkpoint:

```text
models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
```

Restore point:

```text
If proposed v2 fails empirically, use `-ProposedVariant v1` or `scripts/run_model_suite.py full` to return to the v1 comparison path. Do not delete existing ResNet-50 checkpoints or reports.
```

For old v1 progress comparisons, the ResNet-34 wrapper is still available. For the current final-candidate workflow, prefer the ResNet-50 full-v2 comparison documented in `README.md` and `docs/MODEL_COMPARISON_RUNBOOK.md`.

## Encoder Detail

The paper uses `MobileNetV3_Large` as the compression network and transfers ImageNet-1K pretrained weights.

Important paper detail:

```text
layers 1st, 3rd, 6th, and 12th of MobileNetV3_Large are chosen as encoder0 to encoder3 outputs,
and layer 15 is connected to the expansion path after up-sampling.
```

Implementation caution:

The exact layer index mapping must be verified against `torchvision.models.mobilenet_v3_large().features` in the installed torchvision version. Do not assume the paper's layer numbering is identical to torchvision's module indices without printing feature shapes.

Expected behavior for this repo:

```text
input tensor:  [B, 3, H, W]
output logits: [B, 3, H, W]
```

The repo currently uses `num_classes=3`, so the proposed model must return raw logits with 3 channels and must not apply softmax in `forward()`.
Prediction post-processing must use `argmax` over those logits for the three-class segmentation mask.

## PPM Detail

The paper places PPM on the skip connections before the corresponding decoder stages.

PPM uses global average pooling bins:

```text
1x1, 2x2, 3x3, 6x6
```

For each scale, the paper describes:

```text
adaptive/global average pooling -> 1x1 convolution -> bilinear interpolation back to the skip feature size -> channel-wise concatenation
```

Implementation caution:

The paper describes the concept but does not provide every channel projection value. The implementation should keep the PPM output channel count explicit and tested by shape assertions. A practical implementation should use a projection layer after concatenation if needed so each decoder block receives a known skip-channel count.

## Modified Decoder Detail

The decoder block described in the paper is:

```text
concat upsampled decoder feature with PPM skip feature
Conv3x3 + BatchNorm + H-swish
SE attention with reduction ratio 4
Conv3x3 + BatchNorm + H-swish
bilinear upsample
```

SE detail:

```text
global average pooling
FC/1x1 reduction to input_channels / 4
ReLU
FC/1x1 expansion back to input_channels
H-sigmoid
multiply with input feature map
```

The method section and Fig. 6 place SE in the decoder after the first convolution. The conclusion wording says the SE mechanism is mapped into the encoder, but that conflicts with the method section and figure. For implementation, follow the method section and Fig. 6 unless the user explicitly chooses otherwise.

Use PyTorch built-ins where appropriate:

```text
nn.Hardswish
nn.Hardsigmoid
F.interpolate(..., mode="bilinear", align_corners=False)
```

## Paper Training Setup

Paper settings:

| Setting | Value |
| --- | --- |
| Input preprocessing | Normalize and crop to 480x480 |
| Augmentation | Horizontal and vertical flip with probability 0.5 |
| Split | 9:1 train/validation |
| Epochs | 200 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Weight decay | 1e-4 |
| Momentum | 0.9 |
| Batch size | 4 |
| Pretraining | MobileNetV3_Large ImageNet-1K weights from PyTorch |
| Original environment | Ubuntu 20.04, Python 3.8, PyTorch 2.0.0, CUDA 11.8, RTX 3080 10 GB |
| FPS test mentioned in paper | Single RTX 3050 Ti on Windows 11 |

This repo currently trains baselines with Adam and Optuna sampling through `train.py`. If the proposed model must be paper-faithful, a later implementation decision is needed:

```text
Option A: integrate proposed model into the existing Optuna/Adam pipeline for fair comparison with current baselines.
Option B: add a fixed-training mode matching the paper's SGD/lr/momentum/weight-decay settings.
```

For thesis comparison, the more defensible approach is to train proposed and baseline models with the same local budget and report that budget exactly. A separate paper-faithful ablation can be added only if needed.

## Paper Dataset And Metrics

Paper dataset:

```text
1024 cornfield images
classes/species described: common corn, bluegrasses, cirsium setosums, sedges, chenopodium albums
mixed source: field collection in Tai'an, Shandong Province, China plus a public corn weed dataset
public dataset URL mentioned by paper: https://github.com/zhangchuanyin/weed-datasets
```

This is not the same dataset contract as this repo. The architecture can be reused, but class taxonomy and image resolution must follow this repo's data pipeline unless the dataset is intentionally changed.

Paper metrics:

```text
global accuracy
mIoU
Dice coefficient
parameter count
FLOPs
FPS
per-plant IoU
```

This repo now reports these metrics through `evaluate_model_suite.py` and `utils/efficiency.py`.

Local metric policy:

```text
GFLOPs use a 480x480 RGB input.
Conv/Linear multiply-add operations are counted as 2 FLOPs.
FPS and latency use batch size 1.
CPU latency is optional with --cpu_latency.
```

## Reported Paper Results

Ablation table:

| Model | Global accuracy (%) | mIoU (%) | Dice coefficient (%) |
| --- | ---: | ---: | ---: |
| Unet | 99.2 | 84.2 | 81.2 |
| MN_unet | 99.3 | 87.2 | 88.2 |
| MN_PPM_unet | 99.3 | 87.5 | 90.2 |
| MN_PPM_MD_unet | 99.3 | 87.9 | 92.5 |

Model comparison table from the paper:

| Model | Global accuracy (%) | mIoU (%) | Params (M) | FPS | FLOPs (G) |
| --- | ---: | ---: | ---: | ---: | ---: |
| unet | 99.2 | 84.3 | 4.32 | 25.76 | 7.76 |
| MobileNetV3_unet | 99.3 | 87.2 | 3.45 | 81.71 | 0.64 |
| squeezenet_unet | 99.3 | 86.9 | 9.69 | 22.00 | 12.85 |
| convnext_unet | 99.4 | 88.1 | 38.27 | 21.41 | 9.16 |
| efficientnetv2s_unet | 98.9 | 79.3 | 21.73 | 33.50 | 3.94 |
| fcn_resnet50 | 99.2 | 84.7 | 32.95 | 12.16 | 26.58 |
| deeplabv3_resnet50 | 99.2 | 85.2 | 39.64 | 10.87 | 31.41 |
| ours | 99.3 | 87.9 | 3.79 | 58.90 | 0.79 |

Main interpretation:

```text
The final proposed model trades some speed and parameters versus plain MobileNetV3_unet,
but improves mIoU and Dice through PPM plus modified decoder.
```

It does not beat `convnext_unet` on mIoU in the paper, but it is far smaller and faster.

## Paper Limitations

The paper explicitly notes these limitations:

| Limitation | Implementation implication |
| --- | --- |
| Occluded plants are still difficult. | Include qualitative examples with occlusion cases. |
| Dim-light images are still difficult. | Do not claim robust low-light segmentation unless tested locally. |
| Validation was on a single small dataset. | Local dataset results must be reported independently. |
| Weight transfer is not guaranteed across datasets. | Treat ImageNet pretraining as a useful starting point, not proof of generalization. |

## Integration Points For Later Implementation

Implementation files that were touched:

| File | Reason |
| --- | --- |
| `utils/manual_unet_mobilenetv3.py` | Proposed architecture added without changing baseline model internals. |
| `utils/train.py` | New architecture registered in `set_model()`. |
| `utils/parser.py` | New architecture and `mobilenetv3_large` encoder validation added. |
| `predict_testset.py` | Checkpoint filename parsing now supports architecture and encoder names with underscores. |
| `docs/MODEL_COMPARISON_RUNBOOK.md` | Six-model comparison workflow documented. |
| `docs/DEVELOPMENT_TRACKER.md` | Implementation and verification evidence tracked. |
| `AGENTS.md` | Proposed-model contract recorded for future agents. |

Do not change baseline model behavior while adding the proposed model. The safest path is to add a new architecture branch rather than refactor the ResNet baseline implementations.

## Minimum Shape Tests To Run Later

Before long training, run a no-training shape check:

```text
instantiate proposed model with num_classes=3
forward input: torch.randn(2, 3, 480, 480)
assert output shape == (2, 3, 480, 480)
forward input: torch.randn(2, 3, 400, 400) or repo patch size
assert output spatial size matches input
```

Local quick checks used:

```powershell
.\.venv\Scripts\python.exe -m py_compile train.py predict_testset.py small_training.py utils\parser.py utils\train.py utils\manual_unet_mobilenetv3.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
powershell -ExecutionPolicy Bypass -File .\scripts\run_resnet34_architecture_comparison.ps1 -NFolds 2 -MaxEpochs 1 -PlanOnly
```

## Open Decisions Before Coding

| Decision | Recommended default |
| --- | --- |
| Proposed model CLI name | `unet_mobilenetv3`. |
| Encoder positional argument for proposed model | `mobilenetv3_large`; ResNet encoders remain baseline-only. |
| Training optimizer | Existing Adam/Optuna for fair local comparison; add SGD paper-faithful mode only if needed. |
| Input size | Respect repo patch pipeline first; add 480x480 only if dataset preprocessing is changed intentionally. |
| Metrics | Keep current reports, then add Params/FLOPs/FPS once model is stable. |
| Pretrained weights | Use torchvision ImageNet-1K MobileNetV3_Large weights by default; support `--no-pretrained`. |
