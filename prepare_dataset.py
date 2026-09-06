"""
prepare_dataset.py
-------------------
Splits a raw dataset (one folder per breed, containing images) into
train / val / test folders expected by train_model.py.

Expected RAW input structure (you create this by downloading / collecting images):

    raw_dataset/
        Gir/
            img001.jpg
            img002.jpg
            ...
        Murrah/
            img001.jpg
            ...
        Sahiwal/
            ...
        ... (one folder per breed - cattle AND buffalo breeds together)

Output structure created by this script:

    dataset/
        train/<breed>/*.jpg   (70%)
        val/<breed>/*.jpg     (15%)
        test/<breed>/*.jpg    (15%)

Usage:
    python prepare_dataset.py --raw raw_dataset --out dataset --train 0.7 --val 0.15
"""

import argparse
import os
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dataset(raw_dir: str, out_dir: str, train_ratio: float, val_ratio: float, seed: int = 42):
    random.seed(seed)
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset folder '{raw_dir}' not found. Create it with one sub-folder per breed "
            f"(see the docstring at the top of this file / dataset/README.md)."
        )

    breed_dirs = [d for d in raw_path.iterdir() if d.is_dir()]
    if not breed_dirs:
        raise ValueError(f"No breed sub-folders found inside '{raw_dir}'.")

    summary = {}
    for breed_dir in sorted(breed_dirs):
        images = [f for f in breed_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
        if len(images) < 5:
            print(f"  [WARN] '{breed_dir.name}' has only {len(images)} images - consider adding more.")
        random.shuffle(images)

        n_total = len(images)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_test = n_total - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n_total - n_train

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            split_dir = out_path / split_name / breed_dir.name
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, split_dir / f.name)

        summary[breed_dir.name] = {k: len(v) for k, v in splits.items()}
        print(f"  {breed_dir.name:20s} -> train={len(splits['train']):4d}  "
              f"val={len(splits['val']):4d}  test={len(splits['test']):4d}")

    print("\nDone. Dataset written to:", out_path.resolve())
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split raw breed image folders into train/val/test.")
    parser.add_argument("--raw", default="raw_dataset", help="Path to raw dataset (one folder per breed)")
    parser.add_argument("--out", default="dataset", help="Output dataset folder")
    parser.add_argument("--train", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Validation split ratio (remainder -> test)")
    args = parser.parse_args()

    print(f"Splitting '{args.raw}' -> '{args.out}'  (train={args.train}, val={args.val}, "
          f"test={1 - args.train - args.val:.2f})\n")
    split_dataset(args.raw, args.out, args.train, args.val)
