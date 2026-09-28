"""
Explainability for the yield prediction model.
Run:
    python src/yield_explain.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "yield"
DATA_PATH = ROOT / "data" / "unified_yield_dataset.csv"

import sys
sys.path.insert(0, str(ROOT / "src"))
from yield_train import NUMERIC_FEATURES, CATEGORICAL_FEATURES, engineer_features  # noqa: E402


def main():
    pipe = joblib.load(MODEL_DIR / "yield_model.pkl")
    preprocessor = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    sample = df.sample(500, random_state=42)
    X_sample = sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    X_transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    plt.figure()
    shap.summary_plot(
        shap_values, X_transformed, feature_names=feature_names,
        plot_type="bar", max_display=15, show=False,
    )
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "shap_yield_importance.png", dpi=150)
    plt.close()
    print(f"Saved feature importance to {MODEL_DIR}/shap_yield_importance.png")

    # Top features by mean |SHAP value|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:10]
    print("\nTop 10 features driving yield predictions:")
    for i in top_idx:
        print(f"  {feature_names[i]:40s} {mean_abs_shap[i]:.4f}")


if __name__ == "__main__":
    main()

