"""
Command-line crop prediction.

Usage:
    python src/predict.py --N 90 --P 42 --K 43 --temperature 20.9 \
        --humidity 82 --ph 6.5 --rainfall 202.9
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_artifacts():
    model = joblib.load(MODEL_DIR / "crop_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    return model, scaler, encoder


def predict_crop(model, scaler, encoder, values: dict, top_k: int = 3):
    x = np.array([[values[f] for f in FEATURES]])
    x_scaled = scaler.transform(x)

    probs = model.predict_proba(x_scaled)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]

    return [
        {"crop": encoder.classes_[i], "confidence": round(float(probs[i]), 4)}
        for i in top_idx
    ]


def main():
    parser = argparse.ArgumentParser(description="Recommend a crop from soil/climate values")
    for f in FEATURES:
        parser.add_argument(f"--{f}", type=float, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    values = {f: getattr(args, f) for f in FEATURES}
    model, scaler, encoder = load_artifacts()
    results = predict_crop(model, scaler, encoder, values, args.top_k)

    print(json.dumps({"input": values, "recommendations": results}, indent=2))


if __name__ == "__main__":
    main()

