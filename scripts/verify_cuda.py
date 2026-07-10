import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Verify Python, PyTorch, and CUDA visibility.")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Exit successfully even when CUDA is not available.",
    )
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in this environment.")
        return 2

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        if args.allow_cpu:
            return 0
        print("CUDA is required for the RTX training path but is not visible to PyTorch.")
        return 1

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    total_gb = props.total_memory / (1024 ** 3)
    print(f"CUDA device {device_index}: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"VRAM: {total_gb:.1f} GiB")

    x = torch.ones((128, 128), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"CUDA tensor quick check: {float(y[0, 0]):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
