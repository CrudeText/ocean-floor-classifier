# scripts/OceanFloorClassifier.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@Author : CrudeText

Train a CNN on pre-sliced image tiles.
Usage:
  python scripts/OceanFloorClassifier.py \
    --input-dir data/sliced_images \
    --csv-path data/sliced_images/labels.csv \
    --output-dir models \
    --img-size 128 128 \
    --batch-size 32 \
    --epochs 100 \
    --valid-split 0.2
"""
import os
import json
import datetime
import argparse
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger

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
        print("No GPU found, training on CPU.")

def load_labels(csv_path, null_label="NULL"):
    df = pd.read_csv(csv_path)
    df = df[df['label'].notna()]
    df = df[df['label'] != null_label]
    df['label'] = df['label'].astype(str)
    classes = sorted(df['label'].unique())
    label_to_idx = {lbl: idx for idx, lbl in enumerate(classes)}
    return df, classes, label_to_idx

def make_dataset(df, classes, label_to_idx, image_dir, img_size, batch_size, valid_split):
    def load_and_preprocess(name, label):
        path = os.path.join(image_dir, name.numpy().decode())
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = img / 255.0
        return img, tf.one_hot(label, len(classes))

    def tf_parse(name, label):
        img, lab = tf.py_function(load_and_preprocess,
                                  [name, label],
                                  [tf.float32, tf.float32])
        img.set_shape([*img_size, 3])
        lab.set_shape([len(classes)])
        return img, lab

    names = df['tile_name'].tolist()
    labels = [label_to_idx[l] for l in df['label']]
    ds = tf.data.Dataset.from_tensor_slices((names, labels))
    ds = ds.shuffle(len(names)).map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    total_batches = len(names) // batch_size
    val_batches = int(total_batches * valid_split)
    val_ds = ds.take(val_batches)
    train_ds = ds.skip(val_batches)
    return train_ds, val_ds

def build_model(img_size, num_classes):
    m = models.Sequential([
        layers.Input(shape=(*img_size, 3)),
        layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def main():
    parser = argparse.ArgumentParser(description="Train ocean floor classifier CNN")
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory with sliced images')
    parser.add_argument('--csv-path', type=str, required=True,
                        help='CSV file with tile_name,label')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Where to save model run artifacts')
    parser.add_argument('--img-size', type=int, nargs=2, default=[128, 128],
                        help='Image resize dimensions')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--valid-split', type=float, default=0.2)
    args = parser.parse_args()

    setup_gpu()
    df, classes, label_to_idx = load_labels(args.csv_path)
    train_ds, val_ds = make_dataset(df, classes, label_to_idx,
                                     args.input_dir, tuple(args.img_size),
                                     args.batch_size, args.valid_split)

    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    params = {
        'run_id': run_id,
        'input_dir': args.input_dir,
        'csv_path': args.csv_path,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'valid_split': args.valid_split,
        'classes': classes
    }
    with open(os.path.join(run_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    model = build_model(tuple(args.img_size), len(classes))
    model.summary()
    csv_logger = CSVLogger(os.path.join(run_dir, 'training_log.csv'))
    model.fit(train_ds, validation_data=val_ds,
              epochs=args.epochs, callbacks=[csv_logger])
    model.save(os.path.join(run_dir, 'model.h5'))
    print(f"Training complete. Artifacts in {run_dir}")

if __name__ == '__main__':
    main()