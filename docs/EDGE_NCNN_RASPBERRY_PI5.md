# Edge Export Plan: NCNN on Raspberry Pi 5

Status: export tooling is prepared for the final proposed-v2 checkpoint. Raspberry Pi 5 runtime validation is a later phase and must not be claimed until it is measured on the device.

## Target Model

The edge-deployment target is the current proposed-v2 model:

```text
architecture = unet_mobilenetv3_aux
encoder      = mobilenetv3_large
checkpoint   = models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
```

This is a MobileNetV3-Large U-Net with PPM skip/context modules, an SE/H-swish decoder, and an auxiliary foreground head used during training.

The deployment output is only:

```text
segmentation logits: [1, 3, H, W]
```

The auxiliary foreground head is not exported as the primary inference output. It is a training-supervision branch. After training, its effect is already reflected in the shared encoder and decoder weights. Inference still uses the 3-class semantic segmentation logits for Background, Sorghum, and Weed.

## Export Route

Primary route:

```text
PyTorch checkpoint
  -> segmentation-only wrapper
  -> TorchScript model.pt
  -> pnnx
  -> NCNN model.ncnn.param + model.ncnn.bin
```

Fallback route:

```text
PyTorch checkpoint
  -> segmentation-only wrapper
  -> ONNX model.onnx
  -> NCNN conversion tools
```

The segmentation-only wrapper is required because the raw PyTorch model returns a dictionary when the auxiliary foreground head is enabled. Exporting the raw dictionary can produce an ambiguous runtime contract. The wrapper always returns a single tensor:

```text
output = model(input)["segmentation"]
```

## Export Command

Dry-run first:

```powershell
.\.venv\Scripts\python.exe scripts\export_proposed_ncnn.py `
  --checkpoint models\unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar `
  --output_dir exports\ncnn\unet_mobilenetv3_aux_256 `
  --input_size 256 `
  --device cpu `
  --dry_run
```

Run export:

```powershell
.\.venv\Scripts\python.exe scripts\export_proposed_ncnn.py `
  --checkpoint models\unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar `
  --output_dir exports\ncnn\unet_mobilenetv3_aux_256 `
  --input_size 256 `
  --device cpu
```

Expected core package layout:

```text
exports/ncnn/unet_mobilenetv3_aux_256/
  model.pt
  model.ncnn.param
  model.ncnn.bin
  export_manifest.json
  parity_report.json
  README.md
```

`model.onnx` is optional fallback output. It is written only when the local PyTorch ONNX export dependencies are available. If `pnnx`, `onnx`, `onnxscript`, or `onnxruntime` are not installed, the script records the missing route in the manifest/parity report. Do not treat a partial package as a complete NCNN export.

Pnnx can create temporary intermediate files such as `model.pnnx.param`, `model.pnnx.bin`, `model.pnnx.onnx`, `model_pnnx.py`, and `model_ncnn.py`. The exporter removes those by default to keep the deployment folder focused. Use `--keep_intermediates` only when debugging conversion internals.

## Manifest Contract

Always read:

```text
exports/ncnn/unet_mobilenetv3_aux_256/export_manifest.json
exports/ncnn/unet_mobilenetv3_aux_256/parity_report.json
```

The manifest records:

```text
architecture and encoder
checkpoint path and checkpoint SHA256
input shape
output shape
output meaning
class order
mean/std preprocessing
label colors
export route/tool availability
artifact presence
parity status
```

Do not claim NCNN export success unless all of these are present:

```text
model.ncnn.param
model.ncnn.bin
export_manifest.json
parity_report.json
```

and the manifest marks `artifacts.ncnn_exported = true`.

## Preprocessing

Use the same normalization as the training and prediction pipeline:

```text
color order = RGB
scale       = pixel_value / 255.0
mean        = [0.48810686542128406, 0.4733653049842984, 0.4242799605915251]
std         = [0.1321881434144248, 0.12971921686190743, 0.12131885037092494]
```

Do not silently switch to BGR. If the camera or OpenCV path returns BGR, convert to RGB before normalization or document and verify any equivalent channel handling.

## Output Decoding

The model output is raw logits:

```text
logits shape = [1, 3, 256, 256]
```

Decode semantic class ids with:

```text
class_id = argmax(logits, axis=1)
```

Do not use sigmoid thresholding for this 3-class semantic segmentation output.

Class order:

```text
0 = Background
1 = Sorghum
2 = Weed
```

Color map used by repo reports:

```text
Background = [199, 199, 199]
Sorghum    = [31, 119, 180]
Weed       = [255, 127, 14]
```

## Camera Frame Tiling

The first export uses fixed `1x3x256x256`, matching the patch pipeline. For a camera frame:

1. Convert frame to RGB.
2. Split the frame into 256x256 tiles.
3. Pad border tiles as needed.
4. Normalize each tile using the manifest mean/std.
5. Run NCNN inference per tile.
6. Apply argmax per tile.
7. Stitch class-id tiles back to frame layout.
8. Crop padded margins back to the original frame size.

Do not resize an arbitrary frame to 256x256 and stretch the mask back unless that scale change is intentionally evaluated. Resizing can change plant geometry and may degrade class separation.

## Parity Policy

Export correctness is separate from segmentation accuracy.

Minimum parity expectations:

```text
TorchScript or ONNX max_abs_error on logits <= 1e-3
NCNN argmax mask agreement >= 99.5 percent on a real patch
class order unchanged: Background, Sorghum, Weed
```

If raw logits differ slightly but argmax masks are effectively identical, the export can be acceptable. If argmax agreement drops, the NCNN export must be treated as failed even when `.param` and `.bin` files exist.

## Raspberry Pi 5 Runtime Recommendation

Start with NCNN CPU inference on Raspberry Pi 5. Record:

```text
OS and architecture
NCNN build options
thread count
input size
latency ms/tile
FPS at target camera resolution
CPU temperature or throttling notes if relevant
```

Enable Vulkan only after CPU parity is correct. Vulkan support on Pi-class hardware can depend on OS image, driver maturity, and NCNN build flags, so it should be treated as an optimization phase, not the first correctness path.

INT8 quantization is deferred. First make FP32 NCNN export correct, then compare accuracy and latency before introducing quantization.

## Local OpenCV Inference Check

Before writing Raspberry Pi 5 camera code, use the local OpenCV checker to validate preprocessing, tiling, argmax decoding, and visualization behavior.

PyTorch checkpoint GUI mode:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --source 0 --device cuda --view gui
```

PyTorch checkpoint GUI demo with saved video:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --source 0 --device cuda --view gui --max_frames 300 --save_video results\webcam_inference_checks\webcam_cuda_gui_demo.mp4 --run_name webcam_cuda_gui_demo
```

Headless/no-display mode:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --source 0 --device cuda --view headless --max_frames 100
```

Short headless timing check:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --source 0 --device cuda --view headless --max_frames 300 --save_every 100 --run_name webcam_cuda_headless_300
```

Use the headless timing check for local PC-side FPS/latency because GUI drawing and window refresh can add overhead. It is still not a Raspberry Pi 5 benchmark.

Still-image check when no webcam is available:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --image data\test\patches\img\test_p_00000_img.png --device cpu --run_name image_check
```

Optional TorchScript mode:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode torchscript --source 0 --device cpu --view gui --max_frames 100
```

Optional NCNN mode requires a compatible Python `ncnn` binding:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode ncnn --source 0 --view gui --max_frames 100
```

The GUI preview includes original frame, prediction overlay, class mask, runtime status, class coverage, keyboard hint, and the watermark `code by gedxxe`. The GUI is implemented, but it requires a GUI-enabled OpenCV build. If an older venv still uses `opencv-python-headless`, reinstall OpenCV as:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y opencv-python-headless
.\.venv\Scripts\python.exe -m pip install --force-reinstall "opencv-python>=4.10,<5.0"
```

If GUI support is still unavailable, run `--view headless` and inspect saved `preview_*.jpg` files.

For the Windows GUI check, verify:

```powershell
.\.venv\Scripts\python.exe -c "import cv2; print([line.strip() for line in cv2.getBuildInformation().splitlines() if 'GUI:' in line][0])"
```

Expected:

```text
GUI:                           WIN32UI
```

`albumentations` and `albucore` may still report a `pip check` metadata conflict because they declare `opencv-python-headless`. For the GUI checker, the runtime requirement is the opposite: `cv2` must come from GUI-enabled `opencv-python`.

This OpenCV checker is not a Raspberry Pi 5 benchmark. It is a local sanity check. Pi-side FPS/latency must still be measured on the target device.

## References

- NCNN project: https://github.com/Tencent/ncnn
- NCNN PyTorch/ONNX conversion guide: https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx
- NCNN build notes: https://github.com/Tencent/ncnn/wiki/how-to-build
