import argparse
from pathlib import Path


SUBSETS = ("trainval", "test", "test_different_bbch")


def count_files(path, pattern):
    return len(list(path.glob(pattern)))


def main():
    parser = argparse.ArgumentParser(description="Check expected UAV weed dataset folders.")
    parser.add_argument("--root_path", type=str, default=".", help="Repository root containing data/.")
    parser.add_argument(
        "--require-patches",
        action="store_true",
        help="Fail when data/<subset>/patches/img and patches/msk are missing or empty.",
    )
    args = parser.parse_args()

    root = Path(args.root_path)
    data_path = root / "data"
    failures = []

    for subset in SUBSETS:
        subset_path = data_path / subset
        img_count = count_files(subset_path / "img", "*.jpg")
        msk_count = count_files(subset_path / "msk", "*.png")
        patch_img_count = count_files(subset_path / "patches" / "img", "*.png")
        patch_msk_count = count_files(subset_path / "patches" / "msk", "*.png")

        print(
            f"{subset}: raw_img={img_count}, raw_msk={msk_count}, "
            f"patch_img={patch_img_count}, patch_msk={patch_msk_count}"
        )

        if img_count == 0 or msk_count == 0:
            failures.append(f"{subset}: raw img/msk files are missing")
        if img_count != msk_count:
            failures.append(f"{subset}: raw img/msk counts differ")
        if args.require_patches:
            if patch_img_count == 0 or patch_msk_count == 0:
                failures.append(f"{subset}: patch img/msk files are missing")
            if patch_img_count != patch_msk_count:
                failures.append(f"{subset}: patch img/msk counts differ")

    if failures:
        print("Dataset check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Dataset check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
