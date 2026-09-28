# Agri-Advisor — Modules 1, 2, 3 & 4

**Module 1: Crop Recommendation** — recommends the most suitable crop
given soil nutrients (N, P, K), temperature, humidity, pH, and rainfall.
**Module 2: Yield Prediction** — forecasts expected yield (kg/hectare)
given weather, soil, and farm-management inputs, with a genuine
time-based train/test split. Disease detection and price forecasting are
the remaining modules of the larger Smart Agri-Advisor project.

## Dataset — Module 1

2,200 rows, 22 crop classes (100 samples each — perfectly balanced), 7
numeric features. No missing values. Source: standard Crop Recommendation
Dataset (N, P, K, temperature, humidity, ph, rainfall).

## Dataset — Module 2

75,000 rows, 20 Indian states, 22 crops, 2015–2024. 27 columns covering
weather (rainfall, temperature, humidity, sunshine, GDD, pressure, wind),
soil (pH, quality, organic carbon, N/P/K, moisture), and farm management
(irrigation type, seed variety, fertilizer, pesticide use, crop price).
**Note: this dataset is synthetic** (generated to mirror real agronomic
relationships) rather than measured field data — it's excellent for
learning the ML pipeline end-to-end, but a production system would need
to retrain on real agricultural-survey data before being trusted for
actual farm decisions. Source: github.com/Pushkarjay/Crop-Yield-Prediction

## Project structure

```
agri-advisor/
├── data/
│   ├── crop_recommendation.csv
│   └── unified_yield_dataset.csv
├── models/                       # generated after training
│   ├── crop_model.pkl / scaler.pkl / label_encoder.pkl
│   ├── model_comparison.csv / classification_report.json
│   ├── confusion_matrix.npy / meta.json
│   ├── shap_global_importance.png
│   └── yield/
│       ├── yield_model.pkl
│       ├── group_yield_stats.csv   # per (State, Crop) history for lag features
│       ├── model_comparison.csv / meta.json
│       └── shap_yield_importance.png
├── src/
│   ├── train.py           # Module 1: trains & compares 7 classifiers
│   ├── explain.py          # Module 1: SHAP explainability
│   ├── predict.py          # Module 1: CLI single prediction
│   ├── yield_train.py       # Module 2: feature engineering + regression models
│   ├── yield_explain.py      # Module 2: SHAP explainability
│   ├── yield_predict.py      # Module 2: CLI single prediction
│   ├── price_train.py / price_predict.py       # Module 3: price forecasting
│   ├── disease_common.py / disease_train.py    # Module 4: CNN pipeline (train on GPU)
│   ├── disease_predict.py                      # Module 4: diagnosis + Grad-CAM
│   └── api.py                 # FastAPI service exposing all modules
├── tools/make_synthetic_dataset.py             # CPU smoke-test data for Module 4
├── notebooks/train_disease_colab.ipynb         # one-click GPU training for Module 4
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run — Module 1 (crop recommendation)

```bash
python src/train.py       # trains & compares models, saves the best
python src/explain.py      # SHAP explainability report

python src/predict.py --N 90 --P 42 --K 43 --temperature 20.9 \
    --humidity 82 --ph 6.5 --rainfall 202.9
```

## Run — Module 2 (yield prediction)

```bash
python src/yield_train.py    # feature engineering + model comparison
python src/yield_explain.py   # SHAP explainability report

python src/yield_predict.py --state Maharashtra --crop Rice --season Kharif \
    --rainfall 3223 --temperature 26.7 --humidity 95 --sunshine 9 --gdd 1691 \
    --rainfall_anomaly -0.125 --pressure 99.7 --wind 10.3 \
    --soil_type Laterite --soil_ph 6.23 --soil_quality 49.4 --organic_carbon 2.21 \
    --nitrogen 110 --phosphorus 60 --potassium 16.3 --soil_moisture 23.5 \
    --irrigation Tubewell --seed_variety Hybrid --fertilizer 341 \
    --pesticide Yes --crop_price 2013
```

## Serve both modules as one API

```bash
uvicorn src.api:app --reload --port 8000
# POST http://localhost:8000/predict        (crop recommendation)
# POST http://localhost:8000/predict/yield  (yield prediction)
```

## Results

7 models compared with 5-fold cross-validation:

| Model | Test accuracy | Macro F1 | CV mean accuracy |
|---|---|---|---|
| **Random Forest (chosen)** | **99.55%** | **0.9955** | 99.43% (±0.54%) |
| Gradient Boosting | 98.86% | 0.9887 | 98.75% |
| XGBoost | 98.86% | 0.9886 | 98.98% |
| SVM (RBF) | 98.41% | 0.9840 | 97.61% |
| Decision Tree | 97.95% | 0.9794 | 98.47% |
| KNN | 97.95% | 0.9793 | 96.53% |
| Logistic Regression | 97.27% | 0.9725 | 96.76% |

SHAP analysis shows **humidity, potassium (K), and nitrogen (N)** are the
strongest predictors of crop suitability — consistent with real
agronomic knowledge, which is a useful sanity check that the model has
learned something real rather than a dataset artifact.

## Results — Module 2 (yield prediction)

Evaluated on a **time-based holdout** (trained on 2015–2022, tested on
unseen 2023–2024 — not a random split, since real yield forecasting
never gets to see the future):

| Model | Test R² | RMSE (kg/ha) | MAE (kg/ha) | MAPE |
|---|---|---|---|---|
| **XGBoost (chosen)** | **0.9586** | **3,752.8** | 1,636.0 | **16.92%** |
| Random Forest | 0.9566 | 3,840.5 | 1,680.8 | 18.77% |
| Gradient Boosting | 0.9512 | 4,071.4 | 1,790.0 | 19.31% |
| Linear Regression | 0.8895 | 6,128.7 | 2,440.9 | 25.49% |
| Ridge | 0.8889 | 6,146.5 | 2,443.3 | 25.49% |

3-fold CV R² for XGBoost: 0.979 (±0.0006) — consistent, not a lucky split.

Key engineering choices:
- **Time-based split** (not random) — the only honest way to validate a forecasting model
- **log1p-transformed target** — yield spans 3 orders of magnitude across crops (pulses ~700 kg/ha vs sugarcane ~75,000 kg/ha), so training on raw values would let sugarcane errors dominate the loss
- **Lag + rolling-average features** per (State, Crop) — the single strongest predictor, since it captures each crop's baseline yield level; at inference time (`yield_predict.py`), these are looked up from historical group stats rather than assumed known
- **Interaction terms** (NPK sum, NPK-to-fertilizer ratio, rainfall×temperature)

SHAP confirms this is sane: after the lag/rolling features (which mostly
encode "which crop/region is this"), **rainfall, crop price, fertilizer
amount, soil quality, and rainfed vs. irrigated** are the top real
drivers — exactly what an agronomist would expect.

## Module 3 — Mandi price forecasting (onion, Lasalgaon)

Data: **real** daily mandi reports (NHRDF / data.gov.in), 2020-03 to 2022-07,
resampled to 116 weekly points. Source: github.com/amitkaps/onions-dataset.
Held out the last 16 weeks and compared models on a true multi-step forecast:

| Model | RMSE (Rs/q) | MAPE |
|---|---|---|
| **XGBoost on lag/rolling/seasonal features (chosen)** | 170.9 | **14.2%** |
| Prophet | 231.6 | 19.2% |
| SARIMA | 340.2 | 24.6% |
| Seasonal naive baseline | 481.1 | 39.8% |

Notes on honesty: an early run reported 7.4% MAPE; that was wrong (one
feature leaked the current price and XGBoost was scored one step ahead
while the others forecast blind). Fixed with shifted features and a
recursive multi-step forecast. LSTM was deliberately skipped: ~100
training points is far too little for a neural net. Only ~2.3 years of
history exist, so yearly seasonality is learned from barely two cycles;
treat forecasts as directional, not precise.

The bundled observations end in July 2022 and are not suitable for current
market decisions. `price_predict.py` now refuses to produce a forecast when
the latest observation is more than 90 days old. Refresh the mandi history
and retrain before using this endpoint. Forecast loading follows the model
recorded in `models/price/meta.json` (XGBoost, Prophet, SARIMA, or seasonal
naive).

```bash
python src/price_train.py
python src/price_predict.py --weeks 8 --plot
# GET http://localhost:8000/forecast/price?weeks=8
```

## Module 4 — Leaf disease detection (CNN, transfer learning, Grad-CAM)

**Status: pipeline written and smoke-tested on synthetic images (CPU); NOT yet
trained on real data.** Train it on a GPU (Colab/Kaggle) with
`notebooks/train_disease_colab.ipynb`, or run the script directly. No accuracy
numbers are claimed here because none have been measured on PlantVillage yet.

What the pipeline does (`src/disease_train.py`):
- Stratified train/val/test split; ImageNet-pretrained backbone (EfficientNet-B0 default; MobileNetV3-Large for phones; ResNet-50)
- Phase 1: train only the new classifier head. Phase 2: unfreeze all, fine-tune with cosine LR decay (fresh head at 10x LR)
- Heavy colour/scale/rotation augmentation to fight the lab-to-field gap
- Square-root-inverse-frequency class weights + label smoothing (PlantVillage is imbalanced)
- Best checkpoint and early stopping on validation **macro-F1**
- **Temperature scaling** on the validation set so confidence is meaningful; reports ECE before/after
- Test report: accuracy, top-3, macro-F1, per-class report, confusion matrix + top confusions
- Optional **field-image evaluation** (`--field_dir`, e.g. PlantDoc) to measure the lab-to-field gap
- Exports `.pt` checkpoint, TorchScript `.ts.pt`, `class_names.json`, `metrics.json`

```bash
# GPU (Colab/Kaggle); PlantVillage "color" folder = one sub-folder per class
python src/disease_train.py --data_dir /path/to/plantvillage/color \
    --arch efficientnet_b0 --epochs_head 3 --epochs_finetune 12 --batch_size 64 \
    --field_dir /path/to/PlantDoc-Dataset/test          # optional but recommended

# Diagnose a photo (+ Grad-CAM heatmap of what the model looked at)
python src/disease_predict.py leaf.jpg --gradcam gradcam.png

# API: POST an image to /predict/disease?gradcam=true  (needs models/disease/disease_model.pt)
# CPU smoke test of the code path (proves it runs; the toy data teaches nothing real):
python tools/make_synthetic_dataset.py --out /tmp/synth
python src/disease_train.py --data_dir /tmp/synth/train --no_pretrained --img_size 96 --arch efficientnet_b0 \
    --epochs_head 1 --epochs_finetune 15 --lr_finetune 1e-3 --num_workers 0
```

**Read before trusting the numbers you get:**
- PlantVillage has many near-duplicate photos of the same leaf, so a random split leaks and test accuracy is inflated (often 98-99%+). The field-image score is the honest one; expect a large drop.
- Low-confidence inputs (blurry, not a leaf, unseen disease) return `status: "uncertain"` instead of a guess (`--min_conf`, default 0.6).
- Advice text is deliberately generic ("confirm with your local KVK / extension officer"). Do not add pesticide recommendations without agronomist review; optional per-class text can go in `data/disease_advice.json`.
- Grad-CAM shows where the model looked, not proof it is right: check that heatmaps sit on lesions and not on backgrounds.
- Smoke-test finding: torchvision's MobileNetV3 uses BatchNorm momentum 0.01, which made eval-mode accuracy lag train-mode on short runs; `build_model` sets 0.1.

## What's next

- Train Module 4 on a GPU and record real (lab and field) metrics here
- Live Agmarknet feed for Module 3; real (not synthetic) yield data for Module 2
- Combine all four behind one dashboard (see project architecture diagram)

