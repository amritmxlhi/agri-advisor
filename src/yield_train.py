"""
Module 2: Crop Yield Prediction — Training
--------------------------------------------
Predicts Yield_kg_per_hectare from weather, soil, and farm-management
features. Unlike Module 1 (crop recommendation, a classification problem),
this is regression, and it's temporal: we train on earlier years and test
on the most recent years — a time-based split, not a random one — because
that's how yield prediction actually gets used (forecast the *next*
season, not fill in a random hole in history).

Data: synthetic-but-realistic dataset (75k rows, 20 Indian states, 22
crops, 2015-2024) mirroring real agronomic relationships between weather,
soil, farm inputs and yield. Source: github.com/Pushkarjay/Crop-Yield-Prediction

Run:
    python src/yield_train.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "unified_yield_dataset.csv"
MODEL_DIR = ROOT / "models" / "yield"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Yield_kg_per_hectare"
TEST_YEARS = [2023, 2024]  # time-based holdout, not random
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Rainfall_mm", "Temperature_C", "Humidity", "Sunshine_hours", "GDD",
    "Rainfall_Anomaly", "Pressure_KPa", "Wind_Speed_Kmh", "Soil_pH",
    "Soil_Quality", "OrganicCarbon", "Nitrogen", "Phosphorus", "Potassium",
    "Soil_Moisture", "Fertilizer_Amount_kg_per_hectare", "Crop_Price",
    "NPK_sum", "npk_fertilizer_ratio", "rain_temp_interaction",
    "yield_lag_1", "yield_rolling_3yr",
]
CATEGORICAL_FEATURES = [
    "State", "Season", "Crop", "Soil_Type", "Irrigation_Type",
    "Seed_Variety", "Pesticide_Used",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["State", "Crop", "Year"]).copy()

    # Interaction / derived features — this is the heart of feature engineering
    df["NPK_sum"] = df["Nitrogen"] + df["Phosphorus"] + df["Potassium"]
    df["npk_fertilizer_ratio"] = df["NPK_sum"] / df["Fertilizer_Amount_kg_per_hectare"].replace(0, np.nan)
    df["npk_fertilizer_ratio"] = df["npk_fertilizer_ratio"].fillna(df["npk_fertilizer_ratio"].median())
    df["rain_temp_interaction"] = df["Rainfall_mm"] * df["Temperature_C"]

    # Temporal features: previous year's yield and 3-year rolling average,
    # computed per (State, Crop) group — this is what makes it a real
    # forecasting feature rather than a snapshot regression.
    grp = df.groupby(["State", "Crop"])["Yield_kg_per_hectare"]
    df["yield_lag_1"] = grp.shift(1)
    df["yield_rolling_3yr"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    # Early years in each group have no history — fill with the group's
    # overall mean so we don't lose those rows entirely.
    group_mean = df.groupby(["State", "Crop"])["Yield_kg_per_hectare"].transform("mean")
    df["yield_lag_1"] = df["yield_lag_1"].fillna(group_mean)
    df["yield_rolling_3yr"] = df["yield_rolling_3yr"].fillna(group_mean)

    return df


def time_based_split(df: pd.DataFrame):
    train = df[~df["Year"].isin(TEST_YEARS)]
    test = df[df["Year"].isin(TEST_YEARS)]
    print(f"Train: {len(train)} rows (years {sorted(train['Year'].unique())})")
    print(f"Test:  {len(test)} rows (years {sorted(test['Year'].unique())})")
    return train, test


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_candidate_models():
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08, random_state=RANDOM_STATE
        ),
        "xgboost": XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def evaluate(y_true_log, y_pred_log):
    # Metrics on the original yield scale (kg/ha), not the log scale —
    # that's the number that actually means something to a farmer.
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def main():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    train_df, test_df = time_based_split(df)

    X_train, y_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df[TARGET]
    X_test, y_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test_df[TARGET]

    # Log-transform target: yield spans 3 orders of magnitude (pulses vs
    # sugarcane), so training on raw kg/ha would let sugarcane errors
    # dominate the loss. log1p makes the model treat relative error
    # consistently across crops.
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    preprocessor = build_preprocessor()
    models = get_candidate_models()

    results = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train_log)
        preds_log = pipe.predict(X_test)
        metrics = evaluate(y_test_log, preds_log)

        results.append(
            {
                "model": name,
                "test_r2": round(metrics["r2"], 4),
                "test_rmse_kg_ha": round(metrics["rmse"], 1),
                "test_mae_kg_ha": round(metrics["mae"], 1),
                "test_mape_pct": round(metrics["mape"], 2),
            }
        )
        fitted_pipelines[name] = pipe
        print(
            f"{name:20s} | R2={metrics['r2']:.4f} | RMSE={metrics['rmse']:.1f} kg/ha "
            f"| MAPE={metrics['mape']:.2f}%"
        )

    results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)
    print("\n=== Model comparison (sorted by test R2, time-based holdout) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_pipe = fitted_pipelines[best_name]

    # Cross-validate only the winner, to sanity-check the holdout result
    # wasn't a lucky split (cv=3 to keep this fast on 75k rows).
    cv_scores = cross_val_score(best_pipe, X_train, y_train_log, cv=3, scoring="r2", n_jobs=-1)
    print(f"\n{best_name} 3-fold CV R2: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")

    # Save artifacts
    joblib.dump(best_pipe, MODEL_DIR / "yield_model.pkl")
    results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    # Save per-(State, Crop) historical stats so predict.py can build the
    # lag/rolling features for a brand-new prediction request.
    group_stats = (
        df.groupby(["State", "Crop"])["Yield_kg_per_hectare"]
        .agg(["mean", "last"])
        .reset_index()
    )
    group_stats.to_csv(MODEL_DIR / "group_yield_stats.csv", index=False)

    meta = {
        "best_model": best_name,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "target_transform": "log1p",
        "test_years": TEST_YEARS,
        "test_r2": float(results_df.iloc[0]["test_r2"]),
        "test_mape_pct": float(results_df.iloc[0]["test_mape_pct"]),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved best model ({best_name}) and artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

