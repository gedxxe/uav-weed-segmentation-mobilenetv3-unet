# Evaluation Summary

| Model | mIoU | Mean Dice | Pixel Acc | Trainable Params | GFLOPs | Model MB | FPS | Latency ms/img | Efficiency Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unet_mobilenetv3_aux | 0.8143 | 0.8912 | 0.9939 | 3980757 | 17.4793 | 15.2792 | 47.6403 | 20.9906 |  |

Metric notes:

- Segmentation metrics are pixel-level metrics computed from saved prediction masks versus dataset ground truth.
- GFLOPs use a 480x480 RGB input. Conv/Linear multiply-add operations are counted as 2 FLOPs.
- FPS and latency are single-image inference measurements with batch size 1.
- CPU latency is filled only when `--cpu_latency` is used.
