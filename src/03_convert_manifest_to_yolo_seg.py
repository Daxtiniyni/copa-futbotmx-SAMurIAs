import json
import random
import shutil
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np
from project_paths import PROJECT_ROOT

SAM_MASKS_DIR = PROJECT_ROOT / "sam_masks"
MANIFEST_PATH = SAM_MASKS_DIR / "manifest_sam_masks.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_seg"

CLASS_MAP = {
    "ball": 0,
    "goal": 1,
    "robot": 2,
}

TRAIN_RATIO = 0.8
random.seed(42)


def mask_to_yolo_polygon(mask_path, image_width, image_height):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return None

    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 10:
        return None

    epsilon = 0.002 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)

    points = contour.reshape(-1, 2)

    if len(points) < 3:
        return None

    normalized = []

    for x, y in points:
        normalized.append(x / image_width)
        normalized.append(y / image_height)

    return normalized


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"No encontré: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    images = manifest["images"]

    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for split_name, split_images in [
        ("train", train_images),
        ("val", val_images)
    ]:
        image_out_dir = OUTPUT_DIR / "images" / split_name
        label_out_dir = OUTPUT_DIR / "labels" / split_name

        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for item in split_images:
            source_image_path = Path(item["source_image_path"])

            if not source_image_path.exists():
                alt_path = SAM_MASKS_DIR / "images" / item["file_name"]
                source_image_path = alt_path

            if not source_image_path.exists():
                print(f"No encontré imagen: {item['file_name']}")
                continue

            image_width = item["width"]
            image_height = item["height"]

            out_image_path = image_out_dir / item["file_name"]
            out_label_path = label_out_dir / f"{Path(item['file_name']).stem}.txt"

            shutil.copy2(source_image_path, out_image_path)

            lines = []

            for inst in item["instances"]:
                class_name = inst["class_name"]

                if class_name not in CLASS_MAP:
                    continue

                class_id = CLASS_MAP[class_name]
                legacy_mask_path = PureWindowsPath(inst["mask_path"])
                mask_path = (
                    SAM_MASKS_DIR
                    / "masks_instances"
                    / legacy_mask_path.parent.name
                    / legacy_mask_path.name
                )

                if not mask_path.exists():
                    print(f"No encontré máscara: {mask_path}")
                    continue

                polygon = mask_to_yolo_polygon(mask_path, image_width, image_height)

                if polygon is None:
                    continue

                values = [str(class_id)] + [f"{v:.6f}" for v in polygon]
                lines.append(" ".join(values))

            with open(out_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    yaml_path = OUTPUT_DIR / "dataset.yaml"

    yaml_text = """path: .
train: images/train
val: images/val

names:
  0: ball
  1: goal
  2: robot
"""

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print("Conversión terminada")
    print(f"Dataset YOLO guardado en: {OUTPUT_DIR}")
    print(f"Archivo YAML: {yaml_path}")
    print(f"Train: {len(train_images)} imágenes")
    print(f"Val: {len(val_images)} imágenes")


if __name__ == "__main__":
    main()
