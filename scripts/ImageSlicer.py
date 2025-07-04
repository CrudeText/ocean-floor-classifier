# scripts/ImageSlicer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@Author : CrudeText

Tile all images in a folder into fixed-size patches and write a labels CSV stub.
Usage:
  python scripts/ImageSlicer.py \
    --input-dir data/raw_images \
    --output-dir data/sliced_images \
    --tile-size 512 \
    --csv-name labels.csv
"""
import os
import csv
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def slice_image(image_path: Path, output_dir: Path, writer: csv.writer, tile_size: int) -> int:
    img = Image.open(image_path)
    width, height = img.size
    basename = image_path.stem
    count = 0
    for y in range(0, height - tile_size + 1, tile_size):
        for x in range(0, width - tile_size + 1, tile_size):
            tile = img.crop((x, y, x + tile_size, y + tile_size))
            tile_name = f"{basename}_x{x}_y{y}.png"
            tile.save(output_dir / tile_name)
            writer.writerow([tile_name, ""])
            count += 1
    return count

def main():
    p = argparse.ArgumentParser(description="Slice images into tiles and stub labels.csv")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Folder containing raw input images")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory where tiles and CSV will be written")
    p.add_argument("--tile-size", type=int, default=512,
                   help="Size (pixels) of each square tile (default 512)")
    p.add_argument("--csv-name", type=str, default="labels.csv",
                   help="CSV filename to emit with two cols: tile_name,label")
    args = p.parse_args()

    raw_dir = args.input_dir
    out_dir = args.output_dir
    tile_size = args.tile_size
    csv_path = out_dir / args.csv_name

    out_dir.mkdir(parents=True, exist_ok=True)
    image_files = [f for f in os.listdir(raw_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    total = 0
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tile_name", "label"])
        for fname in tqdm(image_files, desc="Slicing images..."):
            in_path = raw_dir / fname
            count = slice_image(in_path, out_dir, writer, tile_size)
            tqdm.write(f"Sliced {fname} -> {count} tiles")
            total += count

    print(f"✅ Done: created {total} tiles in {out_dir}")

if __name__ == '__main__':
    main()
