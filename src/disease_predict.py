"""
Diagnose a leaf photo.

Usage:
    python src/disease_predict.py leaf.jpg --gradcam gradcam.png
    python src/disease_predict.py leaf.jpg --ckpt models/disease/disease_model.pt --topk 3

Confidence is temperature-calibrated. If the top probability is below
--min_conf the result is marked "uncertain" rather than guessing: a blurry
photo, a non-leaf, or a disease outside the training classes should not
produce a confident-sounding diagnosis.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from disease_common import (
    GradCAM, denormalize, eval_transform, get_target_layer, load_bundle, overlay_cam,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "models" / "disease" / "disease_model.pt"
ADVICE_FILE = ROOT / "data" / "disease_advice.json"  # optional: {"Tomato___Late_blight": "..."}

GENERIC_ADVICE = ("Confirm this diagnosis with your local Krishi Vigyan Kendra (KVK) or agriculture "
                  "extension officer before applying any pesticide. Isolate affected plants if possible.")


def parse_label(label: str):
    """PlantVillage style: 'Tomato___Late_blight' -> ('Tomato', 'Late blight', healthy?)."""
    crop, _, cond = label.partition("___")
    cond = cond.replace("_", " ").strip() or "unknown"
    return crop.replace("_", " ").strip(), cond, cond.lower() == "healthy"


def _advice_for(label: str, healthy: bool) -> str:
    if healthy:
        return "No disease detected. Keep monitoring; re-scan if symptoms appear."
    if ADVICE_FILE.exists():
        custom = json.load(open(ADVICE_FILE))
        if label in custom:
            return custom[label]
    return GENERIC_ADVICE


@torch.no_grad()
def _logits(bundle, x):
    return bundle["model"](x.to(bundle["device"]))


def predict_image(img: Image.Image, bundle: dict, topk: int = 3, min_conf: float = 0.6,
                  with_gradcam: bool = False):
    x = eval_transform(bundle["img_size"])(img.convert("RGB")).unsqueeze(0)
    probs = torch.softmax(_logits(bundle, x) / bundle["temperature"], 1)[0].cpu()
    k = min(topk, len(bundle["class_names"]))
    top_p, top_i = probs.topk(k)
    names = bundle["class_names"]

    label = names[int(top_i[0])]
    crop, cond, healthy = parse_label(label)
    conf = float(top_p[0])
    result = {
        "status": "confident" if conf >= min_conf else "uncertain",
        "crop": crop, "condition": cond, "healthy": healthy,
        "confidence": round(conf, 4),
        "top_k": [{"class": names[int(i)], "confidence": round(float(p), 4)} for p, i in zip(top_p, top_i)],
        "advice": (_advice_for(label, healthy) if conf >= min_conf else
                   "Low confidence. Retake the photo: one leaf, in focus, good light, plain background."),
    }
    cam_overlay = None
    if with_gradcam:
        cam_fn = GradCAM(bundle["model"], get_target_layer(bundle["model"], bundle["arch"]))
        try:
            cam, _ = cam_fn(x.to(bundle["device"]), class_idx=int(top_i[0]))
        finally:
            cam_fn.remove()
        cam_overlay = overlay_cam(denormalize(x), cam)
    return result, cam_overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--min_conf", type=float, default=0.6)
    ap.add_argument("--gradcam", default=None, help="save Grad-CAM overlay PNG to this path")
    a = ap.parse_args()

    bundle = load_bundle(a.ckpt)
    result, cam = predict_image(Image.open(a.image), bundle, a.topk, a.min_conf, bool(a.gradcam))
    print(json.dumps(result, indent=2))
    if cam is not None:
        Image.fromarray(cam).save(a.gradcam)
        print(f"Grad-CAM saved to {a.gradcam}")


if __name__ == "__main__":
    main()

