# scripts/InferenceGrid.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author : CrudeText

Perform tiled inference over raw images and save visual overlay results.
Usage:
  python scripts/InferenceGrid.py \
    --raw-dir data/raw_images \
    --model-dir models/run_<timestamp> \
    --output-dir models/run_<timestamp>/inference_results \
    --tile-size 512 \
    --img-size 128 128 \
    --classes rock sand algae mussels
"""
import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found, running on CPU.")

def load_model_and_params(model_dir):
    model = tf.keras.models.load_model(os.path.join(model_dir, 'model.h5'))
    params = json.load(open(os.path.join(model_dir, 'params.json')))
    return model, params['classes']

def parse_args():
    p = argparse.ArgumentParser(description="Grid-based inference over raw images")
    p.add_argument('--raw-dir', type=str, required=True,
                   help="Folder with original JPGs")
    p.add_argument('--model-dir', type=str, required=True,
                   help="Model run directory containing model.h5 + params.json")
    p.add_argument('--output-dir', type=str, required=True,
                   help="Where to save composite inference images")
    p.add_argument('--tile-size', type=int, default=512,
                   help="Tile size in pixels")
    p.add_argument('--img-size', type=int, nargs=2, default=[128,128],
                   help="Model resize dimensions")
    p.add_argument('--classes', nargs='+', required=True,
                   help="List of class names in order")
    args = p.parse_args()
    return args

def main():
    args = parse_args()
    setup_gpu()
    model, classes = load_model_and_params(args.model_dir)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for fname in os.listdir(args.raw_dir):
        if not fname.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        rgb = Image.open(os.path.join(args.raw_dir, fname)).convert('RGB')
        overlay = Image.new('RGBA', rgb.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        w,h = rgb.size
        for y in range(0, h, args.tile_size):
            for x in range(0, w, args.tile_size):
                box = (x, y, min(x+args.tile_size, w), min(y+args.tile_size, h))
                tile = rgb.crop(box).resize(tuple(args.img_size))
                arr = np.array(tile)/255.0
                pred = model.predict(arr[np.newaxis,...], verbose=0)
                cls = classes[np.argmax(pred)]
                color = tuple(params['class_colors'].get(cls, (255,255,255,128)))
                draw.rectangle(box, fill=color, outline=(0,0,0,200))
                bbox = draw.textbbox((0,0), cls, font=font)
                tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                cx, cy = x + (box[2]-box[0])//2, y+(box[3]-box[1])//2
                draw.text((cx-tw/2, cy-th/2), cls, fill=(255,255,255,255), font=font)
        comp = Image.alpha_composite(rgb.convert('RGBA'), overlay)
        comp.convert('RGB').save(os.path.join(out_dir, fname))

    print(f"✅ Inference done. See results in {out_dir}")

if __name__ == '__main__':
    main()
