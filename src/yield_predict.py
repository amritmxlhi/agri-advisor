"""
Predict crop yield for a given state, crop, and season's conditions.

Handles the tricky part of using a time-series feature (yield_lag_1,
yield_rolling_3yr) at inference time: we don't know next season's yield
yet (that's the point), so we look up this State+Crop's historical
average/most-recent yield as a stand-in — exactly what an agronomist
would do by hand ("last time we planted rice here, we got X").

Usage:
    python src/yield_predict.py --state Maharashtra --crop Rice \
        --season Kharif --rainfall 3200 --temperature 27 --humidity 85 \
        --sunshine 8 --gdd 1700 --rainfall_anomaly -0.1 --pressure 100 \
        --wind 10 --soil_type Laterite --soil_ph 6.2 --soil_quality 50 \
        --organic_carbon 2.2 --nitrogen 110 --phosphorus 60 --potassium 16 \
        --soil_moisture 23 --irrigation Tubewell --seed_variety Hybrid \
        --fertilizer 340 --pesticide Yes --crop_price 2000
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "yield"
sys.path.insert(0, str(ROOT / "src"))
from yield_train import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402


def load_artifacts():
    pipe = joblib.load(MODEL_DIR / "yield_model.pkl")
    group_stats = pd.read_csv(MODEL_DIR / "group_yield_stats.csv")
    return pipe, group_stats


def lookup_history(group_stats, state, crop):
    row = group_stats[(group_stats.State == state) & (group_stats.Crop == crop)]
    if row.empty:
        # No history for this exact combo — fall back to the crop's
        # overall average across all states rather than failing.
        row = group_stats[group_stats.Crop == crop]
        if row.empty:
            return None, None
    return float(row.iloc[0]["mean"]), float(row.iloc[0]["last"])


def build_request_row(args, group_stats):
    yield_mean, yield_last = lookup_history(group_stats, args.state, args.crop)
    if yield_mean is None:
        raise ValueError(f"No historical data for crop '{args.crop}' — check spelling.")

    npk_sum = args.nitrogen + args.phosphorus + args.potassium
    npk_fert_ratio = npk_sum / args.fertilizer if args.fertilizer else 0

    row = {
        "Rainfall_mm": args.rainfall,
        "Temperature_C": args.temperature,
        "Humidity": args.humidity,
        "Sunshine_hours": args.sunshine,
        "GDD": args.gdd,
        "Rainfall_Anomaly": args.rainfall_anomaly,
        "Pressure_KPa": args.pressure,
        "Wind_Speed_Kmh": args.wind,
        "Soil_pH": args.soil_ph,
        "Soil_Quality": args.soil_quality,
        "OrganicCarbon": args.organic_carbon,
        "Nitrogen": args.nitrogen,
        "Phosphorus": args.phosphorus,
        "Potassium": args.potassium,
        "Soil_Moisture": args.soil_moisture,
        "Fertilizer_Amount_kg_per_hectare": args.fertilizer,
        "Crop_Price": args.crop_price,
        "NPK_sum": npk_sum,
        "npk_fertilizer_ratio": npk_fert_ratio,
        "rain_temp_interaction": args.rainfall * args.temperature,
        "yield_lag_1": yield_last,
        "yield_rolling_3yr": yield_mean,
        "State": args.state,
        "Season": args.season,
        "Crop": args.crop,
        "Soil_Type": args.soil_type,
        "Irrigation_Type": args.irrigation,
        "Seed_Variety": args.seed_variety,
        "Pesticide_Used": args.pesticide,
    }
    return pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def main():
    p = argparse.ArgumentParser(description="Predict crop yield (kg/hectare)")
    p.add_argument("--state", required=True)
    p.add_argument("--crop", required=True)
    p.add_argument("--season", required=True, choices=["Kharif", "Rabi", "Summer", "Annual"])
    p.add_argument("--rainfall", type=float, required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--humidity", type=float, required=True)
    p.add_argument("--sunshine", type=float, required=True)
    p.add_argument("--gdd", type=float, required=True)
    p.add_argument("--rainfall_anomaly", type=float, default=0.0)
    p.add_argument("--pressure", type=float, required=True)
    p.add_argument("--wind", type=float, required=True)
    p.add_argument("--soil_type", required=True)
    p.add_argument("--soil_ph", type=float, required=True)
    p.add_argument("--soil_quality", type=float, required=True)
    p.add_argument("--organic_carbon", type=float, required=True)
    p.add_argument("--nitrogen", type=float, required=True)
    p.add_argument("--phosphorus", type=float, required=True)
    p.add_argument("--potassium", type=float, required=True)
    p.add_argument("--soil_moisture", type=float, required=True)
    p.add_argument("--irrigation", required=True)
    p.add_argument("--seed_variety", required=True)
    p.add_argument("--fertilizer", type=float, required=True)
    p.add_argument("--pesticide", required=True, choices=["Yes", "No"])
    p.add_argument("--crop_price", type=float, required=True)
    args = p.parse_args()

    pipe, group_stats = load_artifacts()
    X = build_request_row(args, group_stats)

    pred_log = pipe.predict(X)[0]
    pred_yield = float(np.expm1(pred_log))

    print(json.dumps({
        "predicted_yield_kg_per_hectare": round(pred_yield, 1),
        "predicted_yield_tonnes_per_hectare": round(pred_yield / 1000, 3),
        "state": args.state,
        "crop": args.crop,
        "season": args.season,
    }, indent=2))


if __name__ == "__main__":
    main()

