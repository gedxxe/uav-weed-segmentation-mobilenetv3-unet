"""Benchmark one final research checkpoint under a shared FLOP/latency policy.

Run all compared models on the same GPU, input size and precision settings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.efficiency import parameter_counts, summarize_model_efficiency
from utils.train import resolve_device, set_model


def profile(root, run_id, input_size, warmup, iterations, device):
    root = Path(root).resolve()
    report = root / "results" / "research_runs" / run_id
    final_summary = json.loads((report / "final_summary.json").read_text(encoding="utf-8"))
    checkpoint_file = root / final_summary["checkpoint"]
    device = resolve_device(device)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    arch, encoder = checkpoint["architecture"], checkpoint["encoder"]
    model = set_model(
        architecture=arch, encoder_name=encoder, pretrained=False,
        b_bilinear=True, replace_stride_with_dilation=False, num_classes=3,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    total_training_params = parameter_counts(model)
    if arch == "unet_mobilenetv3_aux":
        # Remove the training-only auxiliary output head for fair deployment benchmarking.
        model.foreground_head = None
    result = {
        "run_id": run_id, "checkpoint": str(checkpoint_file.relative_to(root).as_posix()),
        "architecture": arch, "encoder": encoder,
        "benchmark_input": f"1x3x{input_size}x{input_size}",
        "training_parameter_counts": total_training_params,
        "inference_only_auxiliary_head_removed": arch == "unet_mobilenetv3_aux",
    }
    result.update(summarize_model_efficiency(
        model, input_size=input_size, device=device,
        warmup_iterations=warmup,
        benchmark_iterations=iterations,
        use_amp=device == "cuda",
        include_cpu_latency=False,
    ))
    result["note"] = (
        "Same custom FLOP policy as utils/efficiency.py; do not compare uncritically "
        "against third-party 0.79 GFLOPs claims with other FLOP conventions."
    )
    output = report / f"efficiency_{input_size}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-size", type=int, default=480)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.input_size < 32 or args.iterations < 1:
        parser.error("Invalid input size or benchmark iteration count")
    profile(args.root, args.run_id, args.input_size, args.warmup, args.iterations, args.device)


if __name__ == "__main__":
    main()
