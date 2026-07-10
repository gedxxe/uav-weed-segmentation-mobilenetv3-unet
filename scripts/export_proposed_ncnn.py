import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predict_testset import parse_model_config, resolve_model_path
from utils.labels import CLASS_LABELS, CLASS_NAMES, LABEL_COLORS
from utils.model_outputs import get_segmentation_logits
from utils.train import get_calculated_means_stds_trainval, resolve_device, set_model


DEFAULT_CHECKPOINT = (
    "models/unet_mobilenetv3_aux_mobilenetv3_large_dil0_bilin1_pre1.pth.tar"
)
DEFAULT_OUTPUT_DIR = "exports/ncnn/unet_mobilenetv3_aux_256"
EXPECTED_ARCHITECTURE = "unet_mobilenetv3_aux"
EXPECTED_ENCODER = "mobilenetv3_large"
NUM_CLASSES = 3
TORCHSCRIPT_TOLERANCE = 1e-3
NCNN_ARGMAX_AGREEMENT_TARGET = 0.995


class SegmentationOnlyWrapper(torch.nn.Module):
    """Expose only the 3-class semantic logits from the proposed-v2 model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return get_segmentation_logits(self.model(x))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export the final proposed-v2 model as a segmentation-only artifact "
            "for NCNN/Raspberry Pi 5 preparation."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--root_path", default=".")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--sample_patch", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_onnx", action="store_true")
    parser.add_argument("--skip_pnnx", action="store_true")
    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        help="Keep pnnx intermediate files such as model.pnnx.param and model_ncnn.py.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when ONNX/NCNN conversion dependencies are missing.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_status():
    return {
        "pnnx_python": importlib.util.find_spec("pnnx") is not None,
        "pnnx_cli": shutil.which("pnnx"),
        "onnx": importlib.util.find_spec("onnx") is not None,
        "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
        "ncnn_python": importlib.util.find_spec("ncnn") is not None,
    }


def load_checkpoint_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return strip_parallel_prefix(candidate)
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return strip_parallel_prefix(checkpoint)
    raise ValueError(
        f"Unsupported checkpoint structure in {checkpoint_path}. "
        "Expected a state_dict or a dict containing state_dict/model_state_dict."
    )


def strip_parallel_prefix(state_dict):
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def build_model(checkpoint_path, device):
    config = parse_model_config(checkpoint_path)
    architecture = config["architecture"]
    encoder_name = config["encoder_name"]
    if architecture != EXPECTED_ARCHITECTURE or encoder_name != EXPECTED_ENCODER:
        raise ValueError(
            "This exporter is intentionally scoped to the final proposed-v2 model. "
            f"Expected {EXPECTED_ARCHITECTURE} {EXPECTED_ENCODER}, got "
            f"{architecture} {encoder_name}."
        )

    model = set_model(
        architecture=architecture,
        encoder_name=encoder_name,
        pretrained=False,
        b_bilinear=config["b_bilinear"],
        replace_stride_with_dilation=config["replace_stride_with_dilation"],
        num_classes=NUM_CLASSES,
    )
    state_dict = load_checkpoint_state(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return SegmentationOnlyWrapper(model).to(device).eval(), config


def make_dummy_input(input_size, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    tensor = torch.rand(
        (1, 3, input_size, input_size),
        generator=generator,
        dtype=torch.float32,
    )
    return tensor.to(device)


def find_sample_patch(root_path):
    candidates = [
        Path(root_path) / "data" / "test" / "patches" / "img",
        Path(root_path) / "data" / "trainval" / "patches" / "img",
    ]
    for folder in candidates:
        if folder.is_dir():
            matches = sorted(folder.glob("*.png"))
            if matches:
                return matches[0]
    return None


def load_normalized_patch(path, input_size, device):
    means, stds = get_calculated_means_stds_trainval()
    image = Image.open(path).convert("RGB")
    if image.size != (input_size, input_size):
        image = image.resize((input_size, input_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(means, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(stds, dtype=np.float32).reshape(1, 1, 3)
    array = (array - mean) / std
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def tensor_argmax_agreement(a, b):
    pred_a = a.argmax(dim=1)
    pred_b = b.argmax(dim=1)
    return float((pred_a == pred_b).float().mean().item())


def compare_tensors(reference, candidate):
    diff = (reference - candidate).abs()
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "argmax_agreement": tensor_argmax_agreement(reference, candidate),
    }


def export_torchscript(wrapper, dummy_input, output_path):
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input, strict=False)
        traced = torch.jit.freeze(traced.eval())
        traced.save(str(output_path))
    return traced


def export_onnx(wrapper, dummy_input, output_path, opset):
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["segmentation_logits"],
        opset_version=opset,
        do_constant_folding=True,
    )


def run_onnxruntime_parity(onnx_path, input_tensor, reference):
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: input_tensor.detach().cpu().numpy()}
    ort_output = session.run(None, ort_inputs)[0]
    candidate = torch.from_numpy(ort_output).to(reference.device)
    return compare_tensors(reference, candidate)


def run_pnnx_python(torchscript_path, dummy_input, output_dir):
    import pnnx

    pnnx.export(
        str(torchscript_path),
        str(output_dir / "model.ncnn.param"),
        str(output_dir / "model.ncnn.bin"),
        (dummy_input.detach().cpu(),),
    )


def run_pnnx_cli(pnnx_exe, torchscript_path, input_size, output_dir):
    command = [
        pnnx_exe,
        str(torchscript_path.resolve()),
        f"inputshape=[1,3,{input_size},{input_size}]",
    ]
    result = subprocess.run(
        command,
        cwd=output_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_ncnn_python_parity(param_path, bin_path, input_tensor, reference):
    import ncnn

    input_array = np.ascontiguousarray(input_tensor.detach().cpu().numpy()[0])
    with ncnn.Net() as net:
        net.load_param(str(param_path))
        net.load_model(str(bin_path))
        with net.create_extractor() as extractor:
            extractor.input("in0", ncnn.Mat(input_array).clone())
            _, output = extractor.extract("out0")
    output_array = np.asarray(output)
    if output_array.ndim == 3:
        output_array = output_array[None, ...]
    candidate = torch.from_numpy(output_array).to(reference.device, dtype=reference.dtype)
    return compare_tensors(reference, candidate)


def cleanup_pnnx_intermediates(output_dir):
    removed = []
    for filename in (
        "model.pnnx.param",
        "model.pnnx.bin",
        "model.pnnx.onnx",
        "model_pnnx.py",
        "model_ncnn.py",
    ):
        path = output_dir / filename
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    return removed


def build_export_readme(manifest):
    class_lines = "\n".join(
        f"- `{idx}` = {name} (`{label}`), RGB color {list(color)}"
        for idx, (name, label, color) in enumerate(
            zip(CLASS_NAMES, CLASS_LABELS, LABEL_COLORS)
        )
    )
    return f"""# Proposed v2 NCNN Export

This directory is generated by `scripts/export_proposed_ncnn.py`.

## Model

```text
architecture = {manifest["architecture"]}
encoder      = {manifest["encoder"]}
checkpoint   = {manifest["checkpoint_path"]}
input shape  = {manifest["input_shape"]}
output shape = {manifest["output_shape"]}
```

The runtime output is raw 3-class segmentation logits. Apply `argmax` over
channel dimension 1 to obtain class ids. Do not apply sigmoid thresholding for
this semantic segmentation output.

## Classes

{class_lines}

## Preprocessing

Use RGB input, convert to float in `[0, 1]`, then normalize:

```text
mean = {manifest["preprocessing"]["mean"]}
std  = {manifest["preprocessing"]["std"]}
```

## Export Status

```text
overall_status = {manifest["overall_status"]}
ncnn_exported  = {manifest["artifacts"]["ncnn_exported"]}
onnx_exported  = {manifest["artifacts"]["onnx_exported"]}
```

Read `export_manifest.json` and `parity_report.json` before using these files
on Raspberry Pi 5. Do not claim NCNN deployment success unless
`model.ncnn.param`, `model.ncnn.bin`, the manifest, and parity report all exist
and the manifest marks the NCNN artifact as exported.
"""


def make_manifest(
    args,
    checkpoint_path,
    config,
    deps,
    artifacts,
    parity,
    status,
    created_files,
):
    means, stds = get_calculated_means_stds_trainval()
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": config["architecture"],
        "encoder": config["encoder_name"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "input_shape": [1, 3, args.input_size, args.input_size],
        "output_shape": [1, NUM_CLASSES, args.input_size, args.input_size],
        "output_meaning": "raw segmentation logits; use argmax over channel dimension 1",
        "auxiliary_foreground_head": {
            "present_in_checkpoint": True,
            "exported_as_primary_output": False,
            "reason": (
                "The auxiliary foreground head is training supervision only. "
                "Its effect is already encoded in the shared encoder/decoder weights."
            ),
        },
        "class_order": [
            {"id": idx, "name": name, "label": label, "color_rgb": list(color)}
            for idx, (name, label, color) in enumerate(
                zip(CLASS_NAMES, CLASS_LABELS, LABEL_COLORS)
            )
        ],
        "preprocessing": {
            "color_order": "RGB",
            "scale": "pixel_value / 255.0",
            "mean": means,
            "std": stds,
            "input_size": args.input_size,
            "tiling": "256x256 patches; pad edge tiles and crop stitched output back to original frame size",
        },
        "export_tools": {
            "torchscript": "torch.jit.trace + torch.jit.freeze",
            "pnnx_python_available": deps["pnnx_python"],
            "pnnx_cli_path": deps["pnnx_cli"],
            "onnx_available": deps["onnx"],
            "onnxruntime_available": deps["onnxruntime"],
            "ncnn_python_available": deps["ncnn_python"],
            "primary_route": "PyTorch checkpoint -> segmentation-only TorchScript -> pnnx -> NCNN",
            "fallback_route": "PyTorch checkpoint -> segmentation-only ONNX -> NCNN",
        },
        "acceptance_criteria": {
            "torchscript_or_onnx_max_abs_error_logits": TORCHSCRIPT_TOLERANCE,
            "ncnn_argmax_agreement": NCNN_ARGMAX_AGREEMENT_TARGET,
        },
        "artifacts": artifacts,
        "parity_result": summarize_parity(parity),
        "overall_status": status,
        "created_files": created_files,
    }


def summarize_parity(parity):
    summary = {
        "sample_patch": parity.get("sample_patch"),
        "torchscript": parity.get("torchscript"),
        "onnxruntime": parity.get("onnxruntime"),
        "ncnn": {},
    }
    ncnn_result = parity.get("ncnn", {})
    for key in (
        "status",
        "returncode",
        "artifact_check",
        "argmax_agreement",
        "dummy_input",
        "sample_patch",
        "runtime_parity_error",
    ):
        if key in ncnn_result:
            summary["ncnn"][key] = ncnn_result[key]
    if "pnnx_intermediate_cleanup" in parity:
        summary["pnnx_intermediate_cleanup"] = {
            "status": parity["pnnx_intermediate_cleanup"].get("status"),
            "file_count": len(parity["pnnx_intermediate_cleanup"].get("files", [])),
        }
    return summary


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def main():
    args = parse_args()
    root_path = Path(args.root_path)
    checkpoint_path = resolve_model_path(root_path, args.checkpoint)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root_path / output_dir

    deps = dependency_status()
    print("Export target: proposed v2 segmentation-only NCNN package")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    print(f"Input size: {args.input_size}")
    print(f"Dependency status: {json.dumps(deps, indent=2)}")

    config = parse_model_config(checkpoint_path)
    if config["architecture"] != EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"Expected architecture {EXPECTED_ARCHITECTURE}, got {config['architecture']}."
        )

    if args.dry_run:
        print("Dry run only. No export files were written.")
        return 0

    device = resolve_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper, config = build_model(checkpoint_path, device)

    dummy_input = make_dummy_input(args.input_size, device)
    sample_path = Path(args.sample_patch) if args.sample_patch else find_sample_patch(root_path)
    sample_input = None
    if sample_path and sample_path.is_file():
        sample_input = load_normalized_patch(sample_path, args.input_size, device)

    torchscript_path = output_dir / "model.pt"
    onnx_path = output_dir / "model.onnx"
    ncnn_param_path = output_dir / "model.ncnn.param"
    ncnn_bin_path = output_dir / "model.ncnn.bin"
    manifest_path = output_dir / "export_manifest.json"
    parity_path = output_dir / "parity_report.json"
    readme_path = output_dir / "README.md"

    parity = {
        "sample_patch": str(sample_path) if sample_path else None,
        "torchscript": {"status": "not_run"},
        "onnxruntime": {"status": "not_run"},
        "ncnn": {"status": "not_run"},
    }
    artifacts = {
        "torchscript_exported": False,
        "onnx_exported": False,
        "ncnn_exported": False,
        "pnnx_intermediates_kept": args.keep_intermediates,
        "torchscript_path": str(torchscript_path),
        "onnx_path": str(onnx_path),
        "ncnn_param_path": str(ncnn_param_path),
        "ncnn_bin_path": str(ncnn_bin_path),
    }
    created_files = []

    with torch.no_grad():
        torch_reference_dummy = wrapper(dummy_input)
        if list(torch_reference_dummy.shape) != [1, NUM_CLASSES, args.input_size, args.input_size]:
            raise RuntimeError(
                "Unexpected segmentation output shape: "
                f"{list(torch_reference_dummy.shape)}"
            )

        traced = export_torchscript(wrapper, dummy_input, torchscript_path)
        traced_dummy = traced(dummy_input)
        parity["torchscript"] = {
            "status": "checked",
            "dummy_input": compare_tensors(torch_reference_dummy, traced_dummy),
        }
        if sample_input is not None:
            parity["torchscript"]["sample_patch"] = compare_tensors(
                wrapper(sample_input),
                traced(sample_input),
            )
        artifacts["torchscript_exported"] = torchscript_path.is_file()
        if artifacts["torchscript_exported"]:
            created_files.append(str(torchscript_path))

    if not args.skip_onnx:
        try:
            export_onnx(wrapper, dummy_input, onnx_path, args.opset)
            artifacts["onnx_exported"] = onnx_path.is_file()
            if artifacts["onnx_exported"]:
                created_files.append(str(onnx_path))
            if deps["onnxruntime"]:
                parity["onnxruntime"] = {
                    "status": "checked",
                    "dummy_input": run_onnxruntime_parity(
                        onnx_path,
                        dummy_input,
                        torch_reference_dummy,
                    ),
                }
                if sample_input is not None:
                    parity["onnxruntime"]["sample_patch"] = run_onnxruntime_parity(
                        onnx_path,
                        sample_input,
                        wrapper(sample_input),
                    )
            else:
                parity["onnxruntime"] = {
                    "status": "blocked_missing_dependency",
                    "missing": "onnxruntime",
                }
        except Exception as exc:  # noqa: BLE001
            parity["onnxruntime"] = {
                "status": "export_failed",
                "error": str(exc),
            }

    if not args.skip_pnnx:
        if deps["pnnx_python"]:
            try:
                run_pnnx_python(torchscript_path, dummy_input, output_dir)
            except Exception as exc:  # noqa: BLE001
                parity["ncnn"] = {"status": "pnnx_python_failed", "error": str(exc)}
        elif deps["pnnx_cli"]:
            result = run_pnnx_cli(deps["pnnx_cli"], torchscript_path, args.input_size, output_dir)
            parity["ncnn"] = {
                "status": "pnnx_cli_finished" if result["returncode"] == 0 else "pnnx_cli_failed",
                "returncode": result["returncode"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
            }
        else:
            parity["ncnn"] = {
                "status": "blocked_missing_dependency",
                "missing": "pnnx Python package or pnnx executable",
            }

    artifacts["ncnn_exported"] = ncnn_param_path.is_file() and ncnn_bin_path.is_file()
    if artifacts["ncnn_exported"]:
        created_files.extend([str(ncnn_param_path), str(ncnn_bin_path)])
        parity["ncnn"]["artifact_check"] = "param_and_bin_present"
        if deps["ncnn_python"]:
            try:
                parity["ncnn"]["dummy_input"] = run_ncnn_python_parity(
                    ncnn_param_path,
                    ncnn_bin_path,
                    dummy_input,
                    torch_reference_dummy,
                )
                if sample_input is not None:
                    parity["ncnn"]["sample_patch"] = run_ncnn_python_parity(
                        ncnn_param_path,
                        ncnn_bin_path,
                        sample_input,
                        wrapper(sample_input),
                    )
            except Exception as exc:  # noqa: BLE001
                parity["ncnn"]["runtime_parity_error"] = str(exc)
        else:
            parity["ncnn"]["argmax_agreement"] = "not_checked_without_ncnn_runtime_binding"
    elif parity["ncnn"]["status"] == "not_run":
        parity["ncnn"] = {"status": "skipped"}

    if not args.keep_intermediates:
        removed_intermediates = cleanup_pnnx_intermediates(output_dir)
        if removed_intermediates:
            parity["pnnx_intermediate_cleanup"] = {
                "status": "removed",
                "files": removed_intermediates,
            }

    torchscript_ok = (
        parity["torchscript"]["status"] == "checked"
        and parity["torchscript"]["dummy_input"]["max_abs_error"] <= TORCHSCRIPT_TOLERANCE
    )
    if artifacts["ncnn_exported"]:
        status = "ncnn_exported_pending_pi_runtime_parity"
    elif torchscript_ok:
        status = "partial_torchscript_exported_ncnn_blocked"
    else:
        status = "failed_torchscript_parity"

    write_json(parity_path, parity)
    created_files.append(str(parity_path))
    manifest_created_files = created_files + [str(manifest_path), str(readme_path)]

    manifest = make_manifest(
        args=args,
        checkpoint_path=checkpoint_path,
        config=config,
        deps=deps,
        artifacts=artifacts,
        parity=parity,
        status=status,
        created_files=manifest_created_files,
    )
    write_json(manifest_path, manifest)
    created_files.append(str(manifest_path))

    readme_path.write_text(build_export_readme(manifest), encoding="utf-8")
    created_files.append(str(readme_path))

    print(f"Export status: {status}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote parity report: {parity_path}")
    if not artifacts["ncnn_exported"]:
        print("NCNN .param/.bin were not produced. Install pnnx/NCNN conversion tools before claiming NCNN export success.")
        if args.strict:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
