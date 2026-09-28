"""
Module 3: Mandi Price Forecasting
-----------------------------------
Forecasts the weekly modal (wholesale) onion price at Lasalgaon market —
India's largest onion market and the price benchmark the rest of the
country's onion trade watches. This is a genuine univariate time-series
problem: unlike Modules 1-2, there's no rich feature table, just price
and arrivals over time, so the whole game is in the temporal structure
(trend, seasonality, autocorrelation).

Data: real daily mandi price/arrival reports (NHRDF via data.gov.in),
2020-03 to 2022-07, resampled to weekly. Source:
github.com/amitkaps/onions-dataset

Models compared:
  - Seasonal naive (baseline: predict this week = same week last year)
  - SARIMA (statsmodels) - classical, explicitly models trend+seasonality
  - Prophet - handles the irregular real-world reporting gracefully
  - XGBoost on lag/rolling features - lets non-linear interactions between
    recent price momentum and season show up

Run:
    python src/price_train.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "price"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MARKET = "Lasalgaon (Unhali)"
TEST_WEEKS = 16  # ~4 months held out — a realistic forecast horizon for a "sell now or wait" decision


def load_weekly_series() -> pd.DataFrame:
    dfs = [pd.read_csv(DATA_DIR / f"onions-{y}.csv") for y in [2020, 2021, 2022]]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    las = df[df["Market"] == MARKET].copy()
    las["Date"] = pd.to_datetime(las["Date"], format="%d/%b/%Y", errors="coerce")
    # A few rows have "(Avg)" suffixed prices (averaged multi-lot days) — strip and parse
    las["Modal Price (Rs/q)"] = pd.to_numeric(
        las["Modal Price (Rs/q)"].astype(str).str.replace(r"\(.*\)", "", regex=True),
        errors="coerce",
    )
    las = las.dropna(subset=["Date", "Modal Price (Rs/q)"]).sort_values("Date").drop_duplicates("Date")
    las = las.set_index("Date")[["Arrival(q)", "Modal Price (Rs/q)"]]

    # Mandis don't report every single day (weekends, holidays, no-arrival days),
    # so daily data is inherently gappy. Weekly aggregation is both more robust
    # and more decision-relevant (a farmer decides by the week, not the day).
    weekly = las.resample("W").mean().interpolate(limit=2).dropna()
    weekly.columns = ["arrivals", "price"]
    return weekly


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_lag_1"] = df["price"].shift(1)
    df["price_lag_2"] = df["price"].shift(2)
    df["price_lag_4"] = df["price"].shift(4)
    df["price_rolling_4"] = df["price"].shift(1).rolling(4).mean()
    df["price_rolling_8"] = df["price"].shift(1).rolling(8).mean()
    df["arrivals_lag_1"] = df["arrivals"].shift(1)
    df["price_pct_change_1"] = df["price"].shift(1).pct_change(1)
    week_of_year = df.index.isocalendar().week.astype(float)
    df["week_sin"] = np.sin(2 * np.pi * week_of_year / 52)
    df["week_cos"] = np.cos(2 * np.pi * week_of_year / 52)
    return df.dropna()


def build_feature_row(prices: list, last_arrivals: float, date: pd.Timestamp) -> dict:
    """Features for one future week, using only information known before it."""
    p = pd.Series(prices)
    week = float(date.isocalendar().week)
    return {
        "price_lag_1": p.iloc[-1],
        "price_lag_2": p.iloc[-2],
        "price_lag_4": p.iloc[-4],
        "price_rolling_4": p.iloc[-4:].mean(),
        "price_rolling_8": p.iloc[-8:].mean(),
        "arrivals_lag_1": last_arrivals,
        "price_pct_change_1": p.iloc[-1] / p.iloc[-2] - 1,
        "week_sin": np.sin(2 * np.pi * week / 52),
        "week_cos": np.cos(2 * np.pi * week / 52),
    }


def recursive_forecast(model, feature_cols, history: pd.DataFrame, dates) -> np.ndarray:
    """Multi-step forecast: each prediction is fed back in as the next lag.
    This is the honest way to compare against SARIMA/Prophet, which also
    forecast the whole horizon without seeing any future actuals."""
    prices = history["price"].tolist()
    last_arrivals = float(history["arrivals"].iloc[-1])  # future arrivals unknown -> hold last value
    preds = []
    for d in dates:
        row = pd.DataFrame([build_feature_row(prices, last_arrivals, d)])[feature_cols]
        yhat = float(model.predict(row)[0])
        preds.append(yhat)
        prices.append(yhat)
    return np.array(preds)


def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{name:20s} | RMSE={rmse:8.1f} | MAE={mae:8.1f} | MAPE={mape:6.2f}%")
    return {"model": name, "rmse": rmse, "mae": mae, "mape": mape}


def main():
    weekly = load_weekly_series()
    weekly.to_csv(MODEL_DIR / "lasalgaon_weekly_price.csv")
    print(f"Loaded {len(weekly)} weekly observations for {MARKET}")
    print(f"Date range: {weekly.index.min().date()} to {weekly.index.max().date()}")

    train = weekly.iloc[:-TEST_WEEKS]
    test = weekly.iloc[-TEST_WEEKS:]
    print(f"Train: {len(train)} weeks | Test: {len(test)} weeks (last {TEST_WEEKS} weeks held out)\n")

    results = []

    # 1. Seasonal-naive baseline: this week's price = the price 52 weeks ago,
    # falling back to last observed value where a year-ago point doesn't exist.
    naive_preds = []
    full_price = weekly["price"]
    for date in test.index:
        year_ago = date - pd.Timedelta(weeks=52)
        if year_ago in full_price.index:
            naive_preds.append(full_price.loc[year_ago])
        else:
            naive_preds.append(train["price"].iloc[-1])
    results.append(evaluate(test["price"].values, np.array(naive_preds), "seasonal_naive"))

    # 2. SARIMA
    sarima_model = SARIMAX(
        train["price"].values, order=(1, 1, 1), seasonal_order=(1, 0, 0, 52),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    sarima_preds = sarima_model.forecast(steps=TEST_WEEKS)
    results.append(evaluate(test["price"].values, np.asarray(sarima_preds), "sarima"))

    # 3. Prophet
    from prophet import Prophet
    prophet_df = train.reset_index()[["Date", "price"]].rename(columns={"Date": "ds", "price": "y"})
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=TEST_WEEKS, freq="W")
    forecast = prophet_model.predict(future)
    prophet_preds = forecast.set_index("ds").loc[test.index, "yhat"].values
    results.append(evaluate(test["price"].values, prophet_preds, "prophet"))

    # 4. XGBoost on engineered lag/rolling/seasonal features
    feat_df = engineer_features(weekly)
    feature_cols = [c for c in feat_df.columns if c not in ["price", "arrivals"]]
    train_feat = feat_df.loc[feat_df.index.isin(train.index)]
    test_feat = feat_df.loc[feat_df.index.isin(test.index)]

    xgb_model = XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    xgb_model.fit(train_feat[feature_cols], train_feat["price"])
    # Fair comparison: recursive 16-week-ahead forecast, no peeking at test actuals
    xgb_preds = recursive_forecast(xgb_model, feature_cols, train, test.index)
    results.append(evaluate(test["price"].values, xgb_preds, "xgboost_lags"))

    # For reference only: one-step-ahead (uses true last week's price each time)
    onestep = xgb_model.predict(test_feat[feature_cols])
    evaluate(test_feat["price"].values, onestep, "xgb_1step (ref)")

    results_df = pd.DataFrame(results).sort_values("mape")
    print("\n=== Model comparison (sorted by MAPE, last 16-week holdout) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    print(f"\nBest model: {best_name}")

    # Save artifacts
    if best_name == "xgboost_lags":
        # Refit on ALL available weeks for deployment (holdout was only for model selection)
        final_model = XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        ).fit(feat_df[feature_cols], feat_df["price"])
        joblib.dump(final_model, MODEL_DIR / "price_model.pkl")
        joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")
    elif best_name == "prophet":
        joblib.dump(prophet_model, MODEL_DIR / "price_model.pkl")
    elif best_name == "sarima":
        sarima_model.save(str(MODEL_DIR / "price_model_sarima.pkl"))
    elif best_name == "seasonal_naive":
        # This model is fully defined by the saved history; no estimator is needed.
        pass

    results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    meta = {
        "market": MARKET,
        "best_model": best_name,
        "test_weeks": TEST_WEEKS,
        "test_mape": float(results_df.iloc[0]["mape"]),
        "test_rmse": float(results_df.iloc[0]["rmse"]),
        "last_observed_date": str(weekly.index.max().date()),
        "last_observed_price": float(weekly["price"].iloc[-1]),
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

