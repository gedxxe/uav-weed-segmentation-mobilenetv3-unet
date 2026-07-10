import time
from contextlib import contextmanager

import torch
import torch.nn as nn

from utils.train import autocast_context


FLOP_POLICY = "Conv/Linear multiply-add operations are counted as 2 FLOPs."


def parameter_counts(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def model_size_mb(model):
    bytes_total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        bytes_total += tensor.numel() * tensor.element_size()
    return bytes_total / (1024 ** 2)


@contextmanager
def _temporary_hooks(modules, hook_fn):
    handles = [module.register_forward_hook(hook_fn) for module in modules]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def count_flops(model, input_size=480, device="cpu", channels=3):
    """Count approximate inference FLOPs for Conv/ConvTranspose/Linear modules."""
    model = model.to(device=device)
    model.eval()
    flops = {"value": 0}

    def hook(module, inputs, output):
        input_tensor = _first_tensor(inputs)
        output_tensor = _first_tensor(output)
        if input_tensor is None or output_tensor is None:
            return

        if isinstance(module, nn.Conv2d):
            out_elements = output_tensor.numel()
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (
                module.in_channels // module.groups
            )
            flops["value"] += int(out_elements * kernel_ops * 2)
            if module.bias is not None:
                flops["value"] += int(out_elements)
        elif isinstance(module, nn.ConvTranspose2d):
            out_elements = output_tensor.numel()
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (
                module.in_channels // module.groups
            )
            flops["value"] += int(out_elements * kernel_ops * 2)
            if module.bias is not None:
                flops["value"] += int(out_elements)
        elif isinstance(module, nn.Linear):
            batch = input_tensor.shape[0] if input_tensor.ndim > 1 else 1
            flops["value"] += int(batch * module.in_features * module.out_features * 2)
            if module.bias is not None:
                flops["value"] += int(batch * module.out_features)

    profiled_modules = [
        module
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))
    ]
    dummy_input = torch.zeros(1, channels, input_size, input_size, device=device)
    with torch.no_grad():
        with _temporary_hooks(profiled_modules, hook):
            model(dummy_input)

    return {
        "flops": int(flops["value"]),
        "gflops": float(flops["value"] / 1e9),
        "flop_policy": FLOP_POLICY,
    }


def benchmark_latency(
    model,
    input_size=480,
    device="cuda",
    channels=3,
    warmup_iterations=10,
    benchmark_iterations=30,
    use_amp=True,
):
    """Measure single-image inference latency and FPS."""
    model = model.to(device=device)
    model.eval()
    dummy_input = torch.zeros(1, channels, input_size, input_size, device=device)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(max(warmup_iterations, 0)):
            with autocast_context(device, use_amp=use_amp):
                model(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(max(benchmark_iterations, 1)):
            with autocast_context(device, use_amp=use_amp):
                model(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = (elapsed / max(benchmark_iterations, 1)) * 1000.0
    result = {
        "latency_ms_per_image": float(latency_ms),
        "fps": float(1000.0 / latency_ms) if latency_ms > 0 else 0.0,
    }
    if device == "cuda":
        result["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    else:
        result["peak_gpu_memory_mb"] = None
    return result


def summarize_model_efficiency(
    model,
    input_size=480,
    device="cuda",
    warmup_iterations=10,
    benchmark_iterations=30,
    use_amp=True,
    include_cpu_latency=False,
):
    summary = {}
    summary.update(parameter_counts(model))
    summary["model_size_mb"] = float(model_size_mb(model))

    flop_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    summary.update(count_flops(model, input_size=input_size, device=flop_device))

    runtime_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    runtime = benchmark_latency(
        model,
        input_size=input_size,
        device=runtime_device,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
        use_amp=use_amp,
    )
    summary.update(runtime)
    summary["benchmark_device"] = runtime_device

    if include_cpu_latency:
        cpu_runtime = benchmark_latency(
            model,
            input_size=input_size,
            device="cpu",
            warmup_iterations=max(1, warmup_iterations // 2),
            benchmark_iterations=max(1, min(benchmark_iterations, 10)),
            use_amp=False,
        )
        summary["cpu_latency_ms_per_image"] = cpu_runtime["latency_ms_per_image"]
        summary["cpu_fps"] = cpu_runtime["fps"]
    else:
        summary["cpu_latency_ms_per_image"] = None
        summary["cpu_fps"] = None

    return summary
