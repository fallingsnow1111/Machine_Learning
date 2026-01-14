"""Train YOLO with custom modules (ASPP/EMA) without editing ultralytics/.

Why this script exists:
- Ultralytics YAML parser (`parse_model`) looks up modules by name in `ultralytics.nn.tasks` globals().
- We register our custom layers into that namespace at runtime, then load the model YAML.

Usage:
  D:/Anaconda3/envs/yolov11/python.exe train_custom_arch.py \
    --model yolo11-dust-p2-aspp-ema.yaml \
    --data Data/Merged/noise11/dataset_merged.yaml \
    --imgsz 64 --epochs 50 --batch 16

Tip:
- For 1024x1024 images, set imgsz=1024 for both train and val.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

import ultralytics.nn.tasks as tasks

from custom_modules import ASPP, EMA


def register_custom_layers() -> None:
    # Make YAML module names resolvable by ultralytics.nn.tasks.parse_model
    tasks.ASPP = ASPP
    tasks.EMA = EMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    register_custom_layers()

    yolo = YOLO(args.model)
    yolo.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
