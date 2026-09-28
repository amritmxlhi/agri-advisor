"""
Crop Recommendation — Model Training
-------------------------------------
Trains and compares several classifiers on soil + climate data to recommend
the most suitable crop. Saves the best model, scaler, and label encoder to
/models so predict.py and api.py can load them later.

Run:
    python src/train.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "crop_recommendation.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df[TARGET].nunique()} crop classes.")
    return df


def prepare_data(df: pd.DataFrame):
    X = df[FEATURES].values
    y_raw = df[TARGET].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder


def get_candidate_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test, encoder):
    results = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)

        results.append(
            {
                "model": name,
                "test_accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "cv_mean_accuracy": round(cv_scores.mean(), 4),
                "cv_std": round(cv_scores.std(), 4),
            }
        )
        fitted[name] = model
        print(
            f"{name:20s} | test_acc={acc:.4f} | macro_f1={f1:.4f} "
            f"| cv={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})"
        )

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    return results_df, fitted


def save_artifacts(best_name, best_model, scaler, encoder, results_df, X_test, y_test):
    joblib.dump(best_model, MODEL_DIR / "crop_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

    results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    preds = best_model.predict(X_test)
    report = classification_report(
        y_test, preds, target_names=encoder.classes_, output_dict=True
    )
    with open(MODEL_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_test, preds)
    np.save(MODEL_DIR / "confusion_matrix.npy", cm)

    meta = {
        "best_model": best_name,
        "features": FEATURES,
        "classes": encoder.classes_.tolist(),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved best model ({best_name}) and artifacts to {MODEL_DIR}/")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, encoder = prepare_data(df)

    models = get_candidate_models()
    results_df, fitted = evaluate_models(models, X_train, X_test, y_train, y_test, encoder)

    print("\n=== Model comparison (sorted by macro F1) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]

    save_artifacts(best_name, best_model, scaler, encoder, results_df, X_test, y_test)


if __name__ == "__main__":
    main()

