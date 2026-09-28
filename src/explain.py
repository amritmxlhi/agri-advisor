"""
Explainability layer for the crop recommender.
Uses SHAP to show which of N, P, K, temperature, humidity, ph, rainfall
drove a given prediction — this is what turns a black-box model into
something a farmer/agronomist can actually trust.

Run:
    python src/explain.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main():
    model = joblib.load(MODEL_DIR / "crop_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    X_scaled = scaler.transform(X)

    # Sample for speed — SHAP on tree ensembles is fast but no need for all 2200 rows
    sample = X_scaled[:300]
    sample_df = pd.DataFrame(sample, columns=FEATURES)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Global feature importance (averaged across all classes)
    plt.figure()
    shap.summary_plot(
        shap_values, sample_df, plot_type="bar", class_names=encoder.classes_, show=False
    )
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "shap_global_importance.png", dpi=150)
    plt.close()
    print(f"Saved global feature importance to {MODEL_DIR}/shap_global_importance.png")

    # Explain a single example prediction end-to-end
    example = X_scaled[0:1]
    pred_class_idx = model.predict(example)[0]
    pred_class = encoder.inverse_transform([pred_class_idx])[0]
    print(f"\nExample row predicted as: {pred_class}")
    print("Raw input:", df[FEATURES].iloc[0].to_dict())


if __name__ == "__main__":
    main()

