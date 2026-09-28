"""
Crop Recommendation API
------------------------
Run:
    uvicorn src.api:app --reload --port 8000

Then POST to /predict:
    {
      "N": 90, "P": 42, "K": 43,
      "temperature": 20.9, "humidity": 82,
      "ph": 6.5, "rainfall": 202.9
    }
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import base64
import io
import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

sys.path.insert(0, str(ROOT / "src"))
from yield_train import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402

app = FastAPI(title="Agri-Advisor API", version="0.4.0")

# Module 1: crop recommendation
model = joblib.load(MODEL_DIR / "crop_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

# Module 2: yield prediction
yield_pipe = joblib.load(MODEL_DIR / "yield" / "yield_model.pkl")
yield_group_stats = pd.read_csv(MODEL_DIR / "yield" / "group_yield_stats.csv")


class SoilInput(BaseModel):
    N: float = Field(..., ge=0, description="Nitrogen content ratio in soil")
    P: float = Field(..., ge=0, description="Phosphorous content ratio in soil")
    K: float = Field(..., ge=0, description="Potassium content ratio in soil")
    temperature: float = Field(..., ge=-50, le=70, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity in %")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    rainfall: float = Field(..., ge=0, description="Rainfall in mm")


class Recommendation(BaseModel):
    crop: str
    confidence: float


class PredictionResponse(BaseModel):
    recommendations: list[Recommendation]


@app.get("/")
def root():
    return {"status": "ok", "message": "Agri-Advisor crop recommendation API"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SoilInput, top_k: int = Query(default=3, ge=1, le=22)):
    x = np.array([[getattr(payload, f) for f in FEATURES]])
    x_scaled = scaler.transform(x)

    probs = model.predict_proba(x_scaled)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]

    recommendations = [
        Recommendation(crop=encoder.classes_[i], confidence=round(float(probs[i]), 4))
        for i in top_idx
    ]
    return PredictionResponse(recommendations=recommendations)


class YieldInput(BaseModel):
    state: str
    crop: str
    season: str
    rainfall_mm: float = Field(..., ge=0)
    temperature_c: float = Field(..., ge=-50, le=70)
    humidity: float = Field(..., ge=0, le=100)
    sunshine_hours: float = Field(..., ge=0, le=24)
    gdd: float = Field(..., ge=0)
    rainfall_anomaly: float = 0.0
    pressure_kpa: float = Field(..., ge=0)
    wind_speed_kmh: float = Field(..., ge=0)
    soil_type: str
    soil_ph: float = Field(..., ge=0, le=14)
    soil_quality: float = Field(..., ge=0)
    organic_carbon: float = Field(..., ge=0)
    nitrogen: float = Field(..., ge=0)
    phosphorus: float = Field(..., ge=0)
    potassium: float = Field(..., ge=0)
    soil_moisture: float = Field(..., ge=0)
    irrigation_type: str
    seed_variety: str
    fertilizer_kg_per_hectare: float = Field(..., ge=0)
    pesticide_used: str
    crop_price: float = Field(..., ge=0)


class YieldResponse(BaseModel):
    predicted_yield_kg_per_hectare: float
    predicted_yield_tonnes_per_hectare: float


@app.post("/predict/yield", response_model=YieldResponse)
def predict_yield(payload: YieldInput):
    hist = yield_group_stats[
        (yield_group_stats.State == payload.state) & (yield_group_stats.Crop == payload.crop)
    ]
    if hist.empty:
        hist = yield_group_stats[yield_group_stats.Crop == payload.crop]
        if hist.empty:
            raise HTTPException(status_code=400, detail=f"Unknown crop '{payload.crop}'")
    yield_mean = float(hist.iloc[0]["mean"])
    yield_last = float(hist.iloc[0]["last"])

    npk_sum = payload.nitrogen + payload.phosphorus + payload.potassium
    npk_ratio = npk_sum / payload.fertilizer_kg_per_hectare if payload.fertilizer_kg_per_hectare else 0

    row = {
        "Rainfall_mm": payload.rainfall_mm,
        "Temperature_C": payload.temperature_c,
        "Humidity": payload.humidity,
        "Sunshine_hours": payload.sunshine_hours,
        "GDD": payload.gdd,
        "Rainfall_Anomaly": payload.rainfall_anomaly,
        "Pressure_KPa": payload.pressure_kpa,
        "Wind_Speed_Kmh": payload.wind_speed_kmh,
        "Soil_pH": payload.soil_ph,
        "Soil_Quality": payload.soil_quality,
        "OrganicCarbon": payload.organic_carbon,
        "Nitrogen": payload.nitrogen,
        "Phosphorus": payload.phosphorus,
        "Potassium": payload.potassium,
        "Soil_Moisture": payload.soil_moisture,
        "Fertilizer_Amount_kg_per_hectare": payload.fertilizer_kg_per_hectare,
        "Crop_Price": payload.crop_price,
        "NPK_sum": npk_sum,
        "npk_fertilizer_ratio": npk_ratio,
        "rain_temp_interaction": payload.rainfall_mm * payload.temperature_c,
        "yield_lag_1": yield_last,
        "yield_rolling_3yr": yield_mean,
        "State": payload.state,
        "Season": payload.season,
        "Crop": payload.crop,
        "Soil_Type": payload.soil_type,
        "Irrigation_Type": payload.irrigation_type,
        "Seed_Variety": payload.seed_variety,
        "Pesticide_Used": payload.pesticide_used,
    }
    X = pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    pred_log = yield_pipe.predict(X)[0]
    pred = float(np.expm1(pred_log))

    return YieldResponse(
        predicted_yield_kg_per_hectare=round(pred, 1),
        predicted_yield_tonnes_per_hectare=round(pred / 1000, 3),
    )


# Module 3: price forecasting
from price_predict import forecast as price_forecast, advice as price_advice  # noqa: E402


@app.get("/forecast/price")
def forecast_price(weeks: int = 8):
    if not 1 <= weeks <= 16:
        raise HTTPException(status_code=400, detail="weeks must be between 1 and 16")
    try:
        fc, hist = price_forecast(weeks)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "market": "Lasalgaon (Unhali)",
        "commodity": "Onion",
        "last_observed_price": float(hist["price"].iloc[-1]),
        "forecast": [
            {"week_ending": str(r.week_ending.date()), "price_rs_per_quintal": float(r.forecast_price_rs_per_quintal)}
            for r in fc.itertuples()
        ],
        "advice": price_advice(float(hist["price"].iloc[-1]), fc),
    }


# Module 4: leaf disease detection (torch is imported lazily so the other
# endpoints keep working on machines without torch or a trained checkpoint)
_disease_bundle = None


def _get_disease_bundle():
    global _disease_bundle
    if _disease_bundle is None:
        ckpt = Path(os.environ.get("DISEASE_CKPT", MODEL_DIR / "disease" / "disease_model.pt"))
        if not ckpt.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Disease model not found at {ckpt}. Train it with src/disease_train.py "
                       "(see README, Module 4) and place disease_model.pt there.",
            )
        from disease_common import load_bundle
        _disease_bundle = load_bundle(ckpt)
    return _disease_bundle


@app.post("/predict/disease")
async def predict_disease(
    file: UploadFile = File(...),
    gradcam: bool = False,
    min_conf: float = Query(default=0.6, ge=0, le=1),
):
    from PIL import Image, UnidentifiedImageError
    from disease_predict import predict_image

    bundle = _get_disease_bundle()
    try:
        contents = await file.read(10 * 1024 * 1024 + 1)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image upload exceeds the 10 MiB limit.")
        img = Image.open(io.BytesIO(contents))
        img.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    result, cam = predict_image(img, bundle, topk=3, min_conf=min_conf, with_gradcam=gradcam)
    if cam is not None:
        buf = io.BytesIO()
        Image.fromarray(cam).save(buf, format="PNG")
        result["gradcam_png_base64"] = base64.b64encode(buf.getvalue()).decode()
    return result

