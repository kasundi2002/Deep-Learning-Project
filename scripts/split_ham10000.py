#!/usr/bin/env python3
"""
split_ham10000.py
Unzip HAM10000, stratify by dx, and create data/train|val|test/<class>/ folders.

Usage (in Colab after mounting Drive):
  %cd /content/drive/MyDrive/SKIN_CANCER_PROJECT
  !python scripts/split_ham10000.py --train 0.8 --val 0.1 --test 0.1 --seed 42 --clean

Inputs expected in: <project>/raw/
  - HAM10000_images_part_1.zip
  - HAM10000_images_part_2.zip
  - HAM10000_metadata.csv
Outputs written to: <project>/data/
"""

import argparse, os, shutil, zipfile, time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def unzip_if_needed(zip_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    if any(dest_dir.glob("*.jpg")):
        print(f"[skip] Images already present in: {dest_dir}")
        return
    assert zip_path.exists(), f"Missing: {zip_path}"
    print(f"[unzip] {zip_path.name} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)


def build_splits(meta_csv: Path, imgs_dir: Path, train_pct: float, val_pct: float, test_pct: float, seed: int):
    assert abs((train_pct + val_pct + test_pct) - 1.0) < 1e-6, "Splits must sum to 1.0"

    print(f"[meta] Reading: {meta_csv}")
    meta = pd.read_csv(meta_csv)

    # Map image_id -> jpg path & keep only available images
    meta["image_path"] = meta["image_id"].apply(lambda x: str(imgs_dir / f"{x}.jpg"))
    meta = meta[meta["image_path"].map(lambda p: Path(p).exists())].copy()

    classes = sorted(meta["dx"].unique())
    print(f"[meta] Rows with images: {len(meta)} | Classes: {classes}")

    # 1) train vs temp (val+test)
    temp_size = 1.0 - train_pct
    train_df, temp_df = train_test_split(
        meta, test_size=temp_size, stratify=meta["dx"], random_state=seed
    )

    # 2) split temp into val/test in requested ratio
    test_rel = test_pct / (val_pct + test_pct)
    val_df, test_df = train_test_split(
        temp_df, test_size=test_rel, stratify=temp_df["dx"], random_state=seed
    )

    print(f"[split] train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df, classes


def materialize_split(df: pd.DataFrame, split_name: str, out_root: Path):
    base = out_root / split_name
    base.mkdir(parents=True, exist_ok=True)
    for cls in sorted(df["dx"].unique()):
        (base / cls).mkdir(parents=True, exist_ok=True)

    print(f"[copy] {split_name}: {len(df)} files")
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"{split_name:>5}", unit="img"):
        src = Path(r["image_path"])
        dst = base / r["dx"] / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


def print_counts(data_dir: Path):
    print("\n[data] Final counts")
    for split in ["train", "val", "test"]:
        base = data_dir / split
        if not base.exists():
            continue
        counts = {cls.name: len(list((base / cls).glob("*"))) for cls in sorted(base.iterdir()) if cls.is_dir()}
        print(f"  {split}: {counts}")


def main():
    p = argparse.ArgumentParser("HAM10000 unzip + stratified split")
    p.add_argument("--project_dir", default=".", help="Project root (default: current dir)")
    p.add_argument("--raw_dir",      default="", help="Raw dir (default: <project_dir>/raw)")
    p.add_argument("--data_dir",     default="", help="Data dir (default: <project_dir>/data)")
    p.add_argument("--extract_dir",  default="", help="Extract dir (default: <project_dir>/ham10000_extracted)")

    p.add_argument("--part1_name", default="HAM10000_images_part_1.zip")
    p.add_argument("--part2_name", default="HAM10000_images_part_2.zip")
    p.add_argument("--meta_name",  default="HAM10000_metadata.csv")

    p.add_argument("--train", type=float, default=0.80)
    p.add_argument("--val",   type=float, default=0.10)
    p.add_argument("--test",  type=float, default=0.10)
    p.add_argument("--seed",  type=int,   default=42)
    p.add_argument("--clean", action="store_true", help="Delete existing train/val/test before copying")

    args = p.parse_args()

    project_dir = Path(args.project_dir).resolve()
    raw_dir     = Path(args.raw_dir or (project_dir / "raw"))
    data_dir    = Path(args.data_dir or (project_dir / "data"))
    extract_dir = Path(args.extract_dir or (project_dir / "ham10000_extracted"))
    imgs_dir    = extract_dir / "HAM10000_images"

    part1 = raw_dir / args.part1_name
    part2 = raw_dir / args.part2_name
    meta_csv = raw_dir / args.meta_name

    t0 = time.time()
    print(f"[info] project_dir = {project_dir}")
    print(f"[info] raw_dir     = {raw_dir}")
    print(f"[info] data_dir    = {data_dir}")
    print(f"[info] extract_dir = {extract_dir}")

    # Ensure locations exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Unzip both parts into one images directory (idempotent)
    unzip_if_needed(part1, imgs_dir)
    unzip_if_needed(part2, imgs_dir)

    # Build splits using metadata
    train_df, val_df, test_df, classes = build_splits(
        meta_csv, imgs_dir, args.train, args.val, args.test, args.seed
    )

    # Optionally clear existing split folders
    if args.clean:
        for split in ["train", "val", "test"]:
            target = data_dir / split
            if target.exists():
                print(f"[clean] Removing {target}")
                shutil.rmtree(target)

    # Materialize
    materialize_split(train_df, "train", data_dir)
    materialize_split(val_df,   "val",   data_dir)
    materialize_split(test_df,  "test",  data_dir)

    print_counts(data_dir)
    print(f"\n[done] Elapsed: {time.time() - t0:.1f}s")
    print(f"[hint] You can delete '{extract_dir}' afterwards to save Drive space (optional).")


if __name__ == "__main__":
    main()

# How to use it in Colab

# # 1) Mount Drive and go to your project
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
# %cd /content/drive/MyDrive/SKIN_CANCER_PROJECT
# # 2) Run the splitter (80/10/10)
# !python scripts/split_ham10000.py --train 0.8 --val 0.1 --test 0.1 --seed 42 --clean
# # Example: ResNet50
# !python members/run_resnet50.py \
#   --data "/content/drive/MyDrive/SKIN_CANCER_PROJECT/data" \
#   --epochs 20 --warmup 3 --unfreeze 10 --batch 32 \
#   --out_dir runs --run_name m2_try1
