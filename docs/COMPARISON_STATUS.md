# Comparison Status: `test`

- Root: `.`
- Proposed variant: `v1`
- Raw images in subset: 3

## Encoder Summary

| Encoder | Checkpoints | Complete prediction sets | Model reports | Combined report |
| --- | ---: | ---: | ---: | --- |
| `resnet18` | 1/6 | 1/6 | 1/6 | no |
| `resnet34` | 6/6 | 6/6 | 6/6 | no |
| `resnet50` | 6/6 | 6/6 | 6/6 | yes |
| `resnet101` | 1/6 | 1/6 | 1/6 | no |

## Model Detail

| Comparison | Model | Checkpoint | DB | Predictions | Model report |
| --- | --- | --- | --- | ---: | --- |
| `resnet18` | `fcn8s_resnet18` | no | no | 0/3 | no |
| `resnet18` | `fcn16s_resnet18` | no | no | 0/3 | no |
| `resnet18` | `fcn32s_resnet18` | no | no | 0/3 | no |
| `resnet18` | `unet_resnet18` | no | no | 0/3 | no |
| `resnet18` | `dlplus_resnet18` | no | no | 0/3 | no |
| `resnet18` | `unet_mobilenetv3` | yes | yes | 3/3 | yes |
| `resnet34` | `fcn8s_resnet34` | yes | yes | 3/3 | yes |
| `resnet34` | `fcn16s_resnet34` | yes | yes | 3/3 | yes |
| `resnet34` | `fcn32s_resnet34` | yes | yes | 3/3 | yes |
| `resnet34` | `unet_resnet34` | yes | yes | 3/3 | yes |
| `resnet34` | `dlplus_resnet34` | yes | yes | 3/3 | yes |
| `resnet34` | `unet_mobilenetv3` | yes | yes | 3/3 | yes |
| `resnet50` | `fcn8s_resnet50` | yes | yes | 3/3 | yes |
| `resnet50` | `fcn16s_resnet50` | yes | yes | 3/3 | yes |
| `resnet50` | `fcn32s_resnet50` | yes | yes | 3/3 | yes |
| `resnet50` | `unet_resnet50` | yes | yes | 3/3 | yes |
| `resnet50` | `dlplus_resnet50` | yes | yes | 3/3 | yes |
| `resnet50` | `unet_mobilenetv3` | yes | yes | 3/3 | yes |
| `resnet101` | `fcn8s_resnet101` | no | no | 0/3 | no |
| `resnet101` | `fcn16s_resnet101` | no | no | 0/3 | no |
| `resnet101` | `fcn32s_resnet101` | no | no | 0/3 | no |
| `resnet101` | `unet_resnet101` | no | no | 0/3 | no |
| `resnet101` | `dlplus_resnet101` | no | no | 0/3 | no |
| `resnet101` | `unet_mobilenetv3` | yes | yes | 3/3 | yes |

## Combined Report Paths

- `resnet18`: `results\reports\test\architecture_comparison_resnet18`
- `resnet34`: `results\reports\test\architecture_comparison_resnet34`
- `resnet50`: `results\reports\test\architecture_comparison_resnet50`
- `resnet101`: `results\reports\test\architecture_comparison_resnet101`
