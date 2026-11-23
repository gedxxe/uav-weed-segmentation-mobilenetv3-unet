"""Create source-traceable, unfiltered UAV patches for leakage-aware CV.

This intentionally writes to data/trainval/research_patches and never touches
legacy patches, checkpoints, or test data. Split *source captures*, not patches.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from skimage import io as skio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.labels import LABEL_COLORS
from utils.patch_utils import image_to_patches

FIELDS = (
    "patch_image", "patch_mask", "source_image", "row", "col",
    "valid_height", "valid_width", "foreground_fraction",
    "train_eligible",
)


def paired_sources(dataset_dir: Path):
    images = {p.stem: p for p in sorted((dataset_dir / "img").glob("*.jpg"))}
    masks = {p.stem: p for p in sorted((dataset_dir / "msk").glob("*.png"))}
    if not images or images.keys() != masks.keys():
        raise ValueError(
            f"Image/mask source IDs differ at {dataset_dir}. "
            f"Images={sorted(images)}, masks={sorted(masks)}"
        )
    return [(key, images[key], masks[key]) for key in sorted(images)]


def encode_mask(raw_mask: np.ndarray) -> np.ndarray:
    if raw_mask.ndim != 3 or raw_mask.shape[2] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA annotated mask, got {raw_mask.shape}")
    rgb = raw_mask[:, :, :3]
    if raw_mask.shape[2] == 4 and not np.all(raw_mask[:, :, 3] == 255):
        raise ValueError("Mask transparency is unsupported; inspect source labels")
    encoded = np.zeros(rgb.shape[:2], dtype=np.uint8)
    matches_any = np.zeros(rgb.shape[:2], dtype=bool)
    for idx, color in enumerate(LABEL_COLORS):
        matched = np.all(rgb == np.asarray(color, dtype=rgb.dtype), axis=2)
        encoded[matched] = idx
        matches_any |= matched
    if not np.all(matches_any):
        bad = np.unique(rgb[~matches_any].reshape(-1, 3), axis=0)[:8].tolist()
        raise ValueError(f"Unknown ground-truth colors {bad}; do not relabel as background")
    return encoded


def prepare_dataset(root: Path, patch_size=256, min_foreground_fraction=0.01, overwrite=False):
    root = Path(root).resolve()
    dataset_dir = root / "data" / "trainval"
    sources = paired_sources(dataset_dir)
    out_dir = dataset_dir / "research_patches"
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; use --overwrite to regenerate")
        shutil.rmtree(out_dir)
    (out_dir / "img").mkdir(parents=True)
    (out_dir / "msk").mkdir(parents=True)

    rows = []
    for source_id, image_path, mask_path in sources:
        image = skio.imread(image_path)
        mask = encode_mask(skio.imread(mask_path))
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image, got {image.shape} for {image_path}")
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask dimensions mismatch: {image_path}, {mask_path}")
        height, width = mask.shape
        tiles_rgb, _ = image_to_patches(image, slc_size=patch_size, b_msk=False)
        tiles_msk, _ = image_to_patches(mask, slc_size=patch_size, b_msk=True)
        nrows, ncols = math.ceil(height / patch_size), math.ceil(width / patch_size)
        if len(tiles_rgb) != nrows * ncols or len(tiles_rgb) != len(tiles_msk):
            raise AssertionError(f"Incorrect patch count for {source_id}")
        for index in range(nrows * ncols):
            row, col = divmod(index, ncols)
            valid_h = min(patch_size, height - row * patch_size)
            valid_w = min(patch_size, width - col * patch_size)
            label_region = tiles_msk[index, :valid_h, :valid_w]
            fraction = float(np.count_nonzero(label_region) / label_region.size)
            basename = f"{source_id}_r{row:03d}_c{col:03d}"
            img_target = out_dir / "img" / f"{basename}_img.png"
            msk_target = out_dir / "msk" / f"{basename}_msk.png"
            skio.imsave(img_target, tiles_rgb[index], check_contrast=False)
            skio.imsave(msk_target, tiles_msk[index], check_contrast=False)
            rows.append({
                "patch_image": img_target.relative_to(root).as_posix(),
                "patch_mask": msk_target.relative_to(root).as_posix(),
                "source_image": source_id,
                "row": row,
                "col": col,
                "valid_height": valid_h,
                "valid_width": valid_w,
                "foreground_fraction": f"{fraction:.10f}",
                "train_eligible": int(fraction >= min_foreground_fraction),
            })
        print(f"{source_id}: {nrows*ncols} patches, all preserved")
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_images": len(sources),
        "total_patches": len(rows),
        "training_eligible_patches": sum(int(r["train_eligible"]) for r in rows),
        "validation_patches": len(rows),
        "patch_size": patch_size,
        "min_foreground_fraction": min_foreground_fraction,
        "manifest": str(manifest.relative_to(root).as_posix()),
        "note": "Validation retains every patch. Training-only filter is applied AFTER grouped split.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--min-foreground-fraction", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.patch_size < 1 or not (0 <= args.min_foreground_fraction <= 1):
        parser.error("Invalid patch size or foreground threshold")
    print(json.dumps(
        prepare_dataset(args.root, args.patch_size, args.min_foreground_fraction, args.overwrite),
        indent=2,
    ))


if __name__ == "__main__":
    main()
