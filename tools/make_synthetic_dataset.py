"""
Generate a tiny synthetic 'leaf' dataset so the pipeline can be smoke-tested
on CPU without downloading PlantVillage. Each class has a distinct colour and
spot pattern. NOT for real training: it only proves the code runs end to end.

    python tools/make_synthetic_dataset.py --out /tmp/synth
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["Tomato___healthy", "Tomato___Late_blight", "Potato___Early_blight", "Potato___healthy"]


def make_image(cls_idx, rng, size=96):
    base = [(60, 160, 60), (110, 120, 50), (140, 100, 40), (40, 140, 90)][cls_idx]
    img = Image.new("RGB", (size, size), tuple(int(c + rng.integers(-20, 20)) for c in base))
    d = ImageDraw.Draw(img)
    for _ in range(cls_idx * 4):  # more spots = "more diseased"
        x, y, r = rng.integers(10, size - 10), rng.integers(10, size - 10), rng.integers(3, 8)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(70, 40, 20))
    arr = np.asarray(img).astype(int) + rng.integers(-12, 12, (size, size, 3))
    return Image.fromarray(arr.clip(0, 255).astype("uint8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_class", type=int, default=40)
    ap.add_argument("--field_per_class", type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for split, n in (("train", a.per_class), ("field", a.field_per_class)):
        for i, c in enumerate(CLASSES):
            d = Path(a.out) / split / c
            d.mkdir(parents=True, exist_ok=True)
            for j in range(n):
                make_image(i, rng).save(d / f"{j:03d}.png")
    print(f"Wrote synthetic dataset to {a.out}/(train|field)")


if __name__ == "__main__":
    main()

