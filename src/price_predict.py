"""
Forecast the next N weeks of onion modal price at Lasalgaon and give a
plain-language sell/hold hint.

Usage:
    python src/price_predict.py --weeks 8
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "price"
sys.path.insert(0, str(ROOT / "src"))
from price_train import recursive_forecast  # noqa: E402

MAX_HISTORY_AGE_DAYS = 90


def forecast(weeks: int = 8) -> pd.DataFrame:
    if not 1 <= weeks <= 16:
        raise ValueError("weeks must be between 1 and 16")
    hist = pd.read_csv(MODEL_DIR / "lasalgaon_weekly_price.csv", index_col=0, parse_dates=True)
    if hist.empty or "price" not in hist:
        raise ValueError("Price history is missing or has no 'price' column; retrain with valid data.")
    age_days = (pd.Timestamp.now().normalize() - hist.index[-1].normalize()).days
    if age_days > MAX_HISTORY_AGE_DAYS:
        raise ValueError(
            f"Price history is {age_days} days old (latest: {hist.index[-1].date()}). "
            "Refresh the mandi dataset and retrain before requesting a current forecast."
        )
    dates = pd.date_range(hist.index[-1] + pd.Timedelta(weeks=1), periods=weeks, freq="W")
    with open(MODEL_DIR / "meta.json", encoding="utf-8") as f:
        best_model = json.load(f).get("best_model")

    if best_model == "xgboost_lags":
        model = joblib.load(MODEL_DIR / "price_model.pkl")
        cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
        preds = recursive_forecast(model, cols, hist, dates)
    elif best_model == "prophet":
        model = joblib.load(MODEL_DIR / "price_model.pkl")
        future = model.predict(pd.DataFrame({"ds": dates}))
        preds = future["yhat"].to_numpy()
    elif best_model == "sarima":
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults
        model = SARIMAXResults.load(str(MODEL_DIR / "price_model_sarima.pkl"))
        preds = model.forecast(steps=weeks)
    elif best_model == "seasonal_naive":
        prices = hist["price"]
        preds = np.array([
            float(prices.loc[d - pd.Timedelta(weeks=52)])
            if d - pd.Timedelta(weeks=52) in prices.index else float(prices.iloc[-1])
            for d in dates
        ])
    else:
        raise ValueError(f"Unsupported or missing best_model in price metadata: {best_model!r}")
    return pd.DataFrame({"week_ending": dates, "forecast_price_rs_per_quintal": preds.round(0)}), hist


def advice(last_price: float, fc: pd.DataFrame) -> str:
    peak = fc["forecast_price_rs_per_quintal"].max()
    change = (peak - last_price) / last_price * 100
    if change > 10:
        return f"HOLD: model sees prices rising ~{change:.0f}% within the horizon (uncertain: typical error ~14%)."
    if change < -10:
        return f"SELL SOON: model sees prices falling; no upside above current level."
    return "NEUTRAL: no strong move expected; decide on storage cost and cash needs."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=8)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    fc, hist = forecast(a.weeks)
    print(fc.to_string(index=False))
    print("\n" + advice(float(hist["price"].iloc[-1]), fc))
    if a.plot:
        plt.figure(figsize=(9, 4))
        plt.plot(hist.index[-40:], hist["price"].iloc[-40:], label="actual")
        plt.plot(fc["week_ending"], fc["forecast_price_rs_per_quintal"], "--", label="forecast")
        plt.ylabel("Rs / quintal"); plt.legend(); plt.title("Lasalgaon onion modal price")
        plt.tight_layout(); plt.savefig(MODEL_DIR / "forecast.png", dpi=130)


if __name__ == "__main__":
    main()

