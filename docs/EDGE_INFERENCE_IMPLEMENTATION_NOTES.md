# Edge Inference Implementation Notes

This file is for future AI agents or implementers who will write Raspberry Pi 5 inference code for the proposed-v2 model. Treat it as a contract, not as a result report.

## Model Identity

Use this model unless the development tracker says a newer final candidate replaced it:

```text
architecture = unet_mobilenetv3_aux
encoder      = mobilenetv3_large
checkpoint   = models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar
export dir   = exports/ncnn/unet_mobilenetv3_aux_256/
```

Do not replace it with `unet_mobilenetv3` v1 unless the user explicitly asks for the older v1 restore path. The v1 and v2 checkpoint names are separate.

## Required Inference Contract

The inference output must be:

```text
segmentation logits [1, 3, H, W]
```

The final class mask is:

```text
argmax(logits, channel_dimension)
```

Class ids must stay:

```text
0 = Background
1 = Sorghum
2 = Weed
```

Do not use sigmoid thresholding. This is not a binary segmentation model.

## Do Not Do These

- Do not export the raw dict output from the PyTorch model.
- Do not treat the auxiliary foreground head as the final classifier.
- Do not expose the foreground head as the main NCNN output for semantic segmentation.
- Do not change the model identity to `unet_mobilenetv3` v1 without documenting it as a restore or ablation run.
- Do not swap RGB and BGR silently.
- Do not change the training mean/std.
- Do not change class order or color order.
- Do not resize whole camera frames to 256x256 as a shortcut for real-time inference.
- Do not remove padding/tiling behavior without a parity test on stitched masks.
- Do not claim Raspberry Pi 5 real-time performance before Pi-side timing exists.
- Do not claim NCNN export success unless `.param`, `.bin`, manifest, and parity report are all present.

## Preprocessing Contract

Input tile:

```text
shape       = 1x3x256x256
color order = RGB
scale       = pixel_value / 255.0
mean        = [0.48810686542128406, 0.4733653049842984, 0.4242799605915251]
std         = [0.1321881434144248, 0.12971921686190743, 0.12131885037092494]
```

If using OpenCV capture, remember that OpenCV commonly returns BGR arrays. Convert to RGB before normalization unless the NCNN input path explicitly compensates and parity proves the result is equivalent.

## Tiling Contract

The exported model is fixed-size at 256x256 for the first deployment package. For camera frames:

```text
frame -> RGB -> 256x256 tiles -> pad border tiles -> normalize -> infer -> argmax -> stitch -> crop padding
```

Keep the tile metadata:

```text
original frame width and height
tile x/y offset
padding amount
input size
class order
```

This metadata is needed to reconstruct masks correctly and to debug edge-frame output.

## Local OpenCV Checker

Use `scripts/check_webcam_inference.py` as the local reference for camera/video preprocessing before implementing Raspberry Pi 5 runtime code.

Supported modes:

```text
pytorch     = loads the proposed-v2 checkpoint and runs the PyTorch model
torchscript = loads exports/ncnn/unet_mobilenetv3_aux_256/model.pt
ncnn        = loads model.ncnn.param/bin if Python ncnn binding is installed
```

The checker must keep the same rules:

```text
RGB conversion
training mean/std
256x256 tiling with border padding
argmax over 3-class logits
stitch and crop to original frame size
```

GUI mode is explicit:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --source 0 --device cuda --view gui
```

For a short local demo with a saved video:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --source 0 --device cuda --view gui --max_frames 300 --save_video results\webcam_inference_checks\webcam_cuda_gui_demo.mp4 --run_name webcam_cuda_gui_demo
```

For a short PC-side timing check without GUI overhead:

```powershell
.\.venv\Scripts\python.exe scripts\check_webcam_inference.py --mode pytorch --source 0 --device cuda --view headless --max_frames 300 --save_every 100 --run_name webcam_cuda_headless_300
```

The GUI should show original frame, prediction overlay, class mask, FPS/latency, tile grid, class coverage, controls, and the watermark `code by gedxxe`. If GUI support is missing in an older venv, remove `opencv-python-headless`, force reinstall GUI-enabled `opencv-python`, or use `--view headless` and inspect the saved `preview_*.jpg` file instead. `albumentations` may still report a `pip check` metadata conflict because it declares `opencv-python-headless`; for GUI inference, verify `cv2.getBuildInformation()` reports `GUI: WIN32UI`.

Do not treat a successful local OpenCV run as proof of Raspberry Pi 5 real-time performance. It only verifies the local inference path and output contract.

## Parity Before Optimization

The first goal is correctness:

```text
TorchScript/ONNX logits max_abs_error <= 1e-3
NCNN argmax agreement >= 99.5 percent on a real patch
```

Only after parity is acceptable should the implementer tune:

```text
NCNN thread count
Vulkan
tile batching
camera resolution
INT8 quantization
```

## Required Documentation Updates

Every runtime or export change must update:

```text
docs/DEVELOPMENT_TRACKER.md
docs/EDGE_NCNN_RASPBERRY_PI5.md
docs/EDGE_INFERENCE_IMPLEMENTATION_NOTES.md
```

If a command was not actually run, mark it as not run. Do not write final-grade claims from planned commands.
