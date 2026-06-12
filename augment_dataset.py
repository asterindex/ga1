"""
augment_dataset.py
==================
Applies Random Erasing (one rectangle, 20-40% of bounding-box area, pixel
noise fill) placed *inside* the object bounding box parsed from XML labels.
Falls back to random placement if no XML or no valid fit.

Saves augmented images to data2/Images/ and copies labels unchanged.

Usage:
    python augment_dataset.py
"""

import random
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from pathlib import Path

SRC_IMAGES = Path("data/Images")
DST_IMAGES = Path("data2/Images")
SRC_LABELS = Path("data/Labels/XML Format")
DST_LABELS = Path("data2/Labels")
SRC_LABELS_ROOT = Path("data/Labels")

MIN_AREA = 0.20
MAX_AREA = 0.40
MIN_RATIO = 0.3
MAX_RATIO = 3.3


def get_object_bbox(xml_path: Path):
    """Return union bounding box (x1,y1,x2,y2) of all objects in XML, or None."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append((
                int(bb.findtext("xmin", 0)),
                int(bb.findtext("ymin", 0)),
                int(bb.findtext("xmax", 0)),
                int(bb.findtext("ymax", 0)),
            ))
        if not boxes:
            return None
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return x1, y1, x2, y2
    except Exception:
        return None


def erase_inside(img_arr: np.ndarray, bbox) -> np.ndarray:
    """Place noise rectangle inside bbox region (20-40% of bbox area)."""
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw < 4 or bh < 4:
        return erase_random_global(img_arr)

    area = bw * bh
    for _ in range(100):
        target_area = random.uniform(MIN_AREA, MAX_AREA) * area
        ratio = random.uniform(MIN_RATIO, MAX_RATIO)
        rh = int(round((target_area * ratio) ** 0.5))
        rw = int(round((target_area / ratio) ** 0.5))
        if rw >= bw or rh >= bh:
            continue
        x = x1 + random.randint(0, bw - rw)
        y = y1 + random.randint(0, bh - rh)
        noise = np.random.randint(0, 256, (rh, rw, 3), dtype=np.uint8)
        result = img_arr.copy()
        result[y:y + rh, x:x + rw] = noise
        return result

    return erase_random_global(img_arr)


def erase_random_global(img_arr: np.ndarray) -> np.ndarray:
    """Fallback: random placement over full image."""
    h, w = img_arr.shape[:2]
    area = h * w
    for _ in range(100):
        target_area = random.uniform(MIN_AREA, MAX_AREA) * area
        ratio = random.uniform(MIN_RATIO, MAX_RATIO)
        rh = int(round((target_area * ratio) ** 0.5))
        rw = int(round((target_area / ratio) ** 0.5))
        if rw >= w or rh >= h:
            continue
        x = random.randint(0, w - rw)
        y = random.randint(0, h - rh)
        noise = np.random.randint(0, 256, (rh, rw, 3), dtype=np.uint8)
        result = img_arr.copy()
        result[y:y + rh, x:x + rw] = noise
        return result
    return img_arr


def main():
    print("=== Dataset Occlusion Augmentation (object-targeted) ===")
    print(f"Source images : {SRC_IMAGES}")
    print(f"Dest images   : {DST_IMAGES}")
    print(f"XML labels    : {SRC_LABELS}")

    DST_IMAGES.mkdir(parents=True, exist_ok=True)

    # Copy all labels unchanged
    if Path("data2/Labels").exists():
        shutil.rmtree("data2/Labels")
    shutil.copytree(SRC_LABELS_ROOT, "data2/Labels")
    print("Labels copied to data2/Labels")

    files = sorted(SRC_IMAGES.glob("*.jpg")) + \
            sorted(SRC_IMAGES.glob("*.jpeg")) + \
            sorted(SRC_IMAGES.glob("*.png"))
    total = len(files)
    print(f"Images to process: {total}\n")

    used_bbox = 0
    used_fallback = 0

    for i, src in enumerate(files, 1):
        try:
            xml_path = SRC_LABELS / (src.stem + ".xml")
            bbox = get_object_bbox(xml_path) if xml_path.exists() else None

            img = Image.open(src).convert("RGB")
            arr = np.array(img)

            if bbox is not None:
                arr = erase_inside(arr, bbox)
                used_bbox += 1
            else:
                arr = erase_random_global(arr)
                used_fallback += 1

            Image.fromarray(arr).save(DST_IMAGES / src.name, quality=95)
        except Exception as e:
            print(f"  WARNING: skipped {src.name}: {e}")

        if i % 500 == 0 or i == total:
            print(f"  {i}/{total} done")

    print(f"\nAll done. Augmented dataset in: data2/")
    print(f"  Object-targeted: {used_bbox} images")
    print(f"  Fallback (global): {used_fallback} images")


if __name__ == "__main__":
    main()
