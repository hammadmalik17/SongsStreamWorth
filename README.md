<div align="center">
<!-- ANIMATED HEADER BANNER -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=1DB954&height=200&section=header&text=SongsStreamWorth&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Can%20a%20song's%20sound%20predict%20its%20streams%3F&descAlignY=62&descAlign=50"/>
<br/>

<!-- BADGES ROW 1 -->
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Live%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

<!-- BADGES ROW 2 -->
[![Spotify](https://img.shields.io/badge/Spotify-953%20Songs-1DB954?style=for-the-badge&logo=spotify&logoColor=white)](https://www.spotify.com)
[![R²](https://img.shields.io/badge/Best%20R²-0.0708-ff6b6b?style=for-the-badge&logo=chart.js&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)]()

<br/>

<p align="center">
  <b>🎧 An end-to-end machine learning system that predicts Spotify stream counts from raw audio features — BPM, danceability, energy, and more.</b><br/>
  <i>And honestly? The findings are more interesting than the predictions.</i>
</p>

<br/>

---

<!-- QUICK LINKS -->
<p align="center">
  <a href="#-project-overview"><b>Overview</b></a> •
  <a href="#%EF%B8%8F-project-structure"><b>Structure</b></a> •
  <a href="#-ml-pipeline"><b>ML Pipeline</b></a> •
  <a href="#-running-the-api"><b>Run the API</b></a> •
  <a href="#-api-usage"><b>API Docs</b></a> •
  <a href="#-key-insight"><b>Key Insight</b></a>
</p>

---

</div>

## 🎯 Project Overview

> **TL;DR:** We trained ML models to predict Spotify streams from audio features. The best model achieved R² ≈ 0.07. That *low* number is actually the story — and documenting it honestly is what separates real data science from vibes-based conclusions.

This project is a complete, production-grade machine learning pipeline that answers one deceptively simple question:

> *Can what a song **sounds like** predict how many times it gets streamed?*

The pipeline covers everything: raw data ingestion → EDA → feature engineering → model training → experiment tracking → serving a live prediction API via FastAPI → Docker containerisation. Every step is reproducible, every experiment is logged, and every assumption is documented.

**The core finding?** Audio features — BPM, danceability, energy, acousticness, valence — explain only **~7% of stream variance**. Artist reach, editorial playlisting, and social media momentum dominate the rest. We didn't have those features. Saying so loudly is part of the point.

<br/>

---

## 🗂️ Project Structure

```
SongsStreamWorth/
│
├── 📄 spotify-2023.csv           # Raw dataset (953 songs, Kaggle)
├── 📓 ofcAgain.ipynb             # Full notebook: EDA → Feature Eng → Modelling
│
├── 🤖 model/
│   └── hit_predictor.pkl         # Best model (Gradient Boosting + PowerTransformer)
│
├── 🚀 app/
│   ├── main.py                   # FastAPI prediction service
│   ├── schema.py                 # Request/response Pydantic models
│   └── requirements.txt          # API dependencies
│
├── 🐳 Dockerfile                 # Containerised API
├── 🐳 docker-compose.yml         # One-command run
│
├── 📊 mlruns/                    # MLflow experiment logs (auto-generated)
│
└── 📖 README.md
```

<br/>

---

## 🔬 ML Pipeline

The pipeline is broken into five stages. Each one feeds into the next. Here's the full story:

<br/>

### Stage 1 — Data Ingestion & Cleaning

```
Source: Kaggle — "Most Streamed Spotify Songs 2023"
Size:   953 songs × ~24 features
```

- Handled encoding issues in raw CSV (non-UTF-8 characters from song titles)
- Removed commas embedded in numeric strings (e.g. `"1,234,567"` → `1234567`)
- Dropped null rows — particularly entries with missing `streams`
- **Crucially:** removed post-release leakage features — playlist counts, chart positions — that a model would never have access to at prediction time

<br/>

### Stage 2 — Feature Engineering

Raw audio features are heavily skewed. We fix that.

| Feature | Transform Applied | Reason |
|---|---|---|
| `bpm` | Yeo-Johnson | Right-skewed distribution |
| `artist_count` | Yeo-Johnson | Most songs = 1 artist |
| `acousticness_%` | Yeo-Johnson | Bimodal skew |
| `speechiness_%` | Yeo-Johnson | Most songs near 0 |
| `liveness_%` | Yeo-Johnson | Outlier-heavy |
| `streams` (target) | Log transform | Reduces 10⁹ variance |
| `key` | Ordinal encode | C=0, C#=1 ... B=11 |
| `mode` | Binary encode | Major=1, Minor=0 |

> ⚠️ The fitted `PowerTransformer` is **saved alongside the model** in `hit_predictor.pkl` to ensure consistent inference — no leakage, no surprises.

<br/>

### Stage 3 — Model Training & Evaluation

Four models. One winner. Here are the full results:

<br/>

<div align="center">

| 🏷️ Model | 📈 R² (Test) | 📉 RMSE | 🔁 CV R² | Status |
|---|---|---|---|---|
| Linear Regression | 0.0326 | 1.4233 | -0.198 | ❌ Underfit |
| Ridge Regression | 0.0326 | 1.4233 | -0.198 | ❌ Same as Linear |
| Random Forest | 0.0685 | 1.3967 | -0.253 | 🔶 Better, not great |
| **Gradient Boosting** | **0.0708** | **1.3950** | -0.348 | ✅ **Best — Saved** |

</div>

<br/>

> 💡 All cross-validated R² scores are **negative** — meaning even the best model struggles to generalise beyond the training set. This is not a modelling failure; it's a data signal. The information needed to predict streams just isn't in audio features alone.

<br/>

### Stage 4 — Feature Importance

```
Feature                  Importance
─────────────────────────────────────
energy_%                 ████████████████████ 0.191
danceability_%           ███████████████████  0.187
bpm                      ███████████████      0.150
valence_%                █████████████        0.134
acousticness_%_yj        ████████████         0.129
```

Energy and danceability are the strongest audio signals — which makes intuitive sense. But even the top feature explains less than 20% of variation within an already weak model.

<br/>

### Stage 5 — Experiment Tracking with MLflow

Every run is logged. Every metric is versioned. Every artefact is saved.

```bash
# Launch the tracking UI
mlflow ui

# View at:
# http://localhost:5000
```

All experiments are stored under: `spotify-hit-prediction`

Tracked per run:
- Hyperparameters
- Train/test R² and RMSE
- Cross-validation scores
- Model artifact (`.pkl`)
- Feature importance values

<br/>

---

## 🚀 Running the API

Three ways to get the prediction service running. Pick your level:

<br/>

### ⚡ Option 1 — Local Python (Fastest)

```bash
# Install dependencies
pip install -r app/requirements.txt

# Start the server
uvicorn app.main:app --reload
```

<br/>

### 🐳 Option 2 — Docker

```bash
# Build the image
docker build -t spotify-predictor .

# Run the container
docker run -p 8000:8000 spotify-predictor
```

<br/>

### 🎯 Option 3 — Docker Compose *(Recommended)*

```bash
docker-compose up --build
```

That's it. One command. The API is live.

<br/>

<div align="center">

| 🌐 Endpoint | 🔗 URL |
|---|---|
| Prediction API | `http://localhost:8000` |
| Interactive Swagger Docs | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |
| MLflow Dashboard | `http://localhost:5000` |

</div>

<br/>

---

## 📡 API Usage

### `POST /predict` — Get a Stream Prediction

Send audio features, get a stream count prediction with honest caveats.

**Request Body:**

```json
{
  "bpm": 120,
  "key": 5,
  "mode": 1,
  "danceability_pct": 75,
  "valence_pct": 60,
  "energy_pct": 80,
  "acousticness_pct": 20,
  "speechiness_pct": 5,
  "liveness_pct": 12,
  "artist_count": 1
}
```

**Response:**

```json
{
  "predicted_streams": 261019019,
  "log_streams": 19.3801,
  "model_used": "GradientBoostingRegressor",
  "note": "Audio features explain ~7% of stream variance. This is a rough signal — not a chart guarantee."
}
```

<br/>

### `GET /health` — Check Service Health

```json
{
  "status": "ok",
  "model": "GradientBoostingRegressor",
  "features": [
    "bpm",
    "key",
    "mode",
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%_yj",
    "speechiness_%_yj",
    "liveness_%_yj",
    "artist_count_yj"
  ]
}
```

<br/>

**Field Reference:**

| Field | Type | Range | Description |
|---|---|---|---|
| `bpm` | int | 60–220 | Beats per minute |
| `key` | int | 0–11 | Musical key (C=0, C#=1 ... B=11) |
| `mode` | int | 0 or 1 | Minor=0, Major=1 |
| `danceability_pct` | int | 0–100 | Danceability score |
| `valence_pct` | int | 0–100 | Musical positivity |
| `energy_pct` | int | 0–100 | Perceived energy |
| `acousticness_pct` | int | 0–100 | Acoustic confidence |
| `speechiness_pct` | int | 0–100 | Presence of spoken words |
| `liveness_pct` | int | 0–100 | Live audience detection |
| `artist_count` | int | 1–8 | Number of contributing artists |

<br/>

---

## 🧰 Tech Stack

<div align="center">

| Layer | Tool | Role |
|---|---|---|
| 📦 Data | pandas, numpy | Loading, cleaning, wrangling |
| 🔢 Math | scipy | Yeo-Johnson transforms |
| 🤖 ML | scikit-learn | All model training & evaluation |
| 📊 Tracking | MLflow | Experiment logging & model registry |
| 🌐 API | FastAPI + Uvicorn | Prediction service |
| 🐳 Infra | Docker + Compose | Containerisation |
| 🐍 Runtime | Python 3.10 | Core environment |

</div>

<br/>

---

## 📊 Key Insight

<div align="center">

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   What predicts a song's stream count?                  │
│                                                         │
│   🎵 Audio features (BPM, energy, danceability...)      │
│      ████░░░░░░░░░░░░░░░░░░░░░░░░  ~7%                  │
│                                                         │
│   🎤 Artist follower count                              │
│      ████████████████████████████  ~40%+ (estimated)    │
│                                                         │
│   📋 Editorial playlist inclusion                       │
│      ███████████████████████████   major driver         │
│                                                         │
│   📱 Social media momentum (TikTok, IG)                 │
│      ███████████████████████████   major driver         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

</div>

> **The honest finding:** Audio features are *weak* predictors of stream count (R² ≈ 0.07). A song's sonic qualities matter far less than who made it and how it was pushed. The negative cross-validated R² confirms the model doesn't generalise — and that's not a bug, it's a real-world result worth stating clearly.

**What would meaningfully improve this model:**

- 🎤 Artist follower count at time of release
- 📋 Spotify editorial playlist inclusion flag
- 📱 Social momentum signals (TikTok uses, Instagram mentions)
- 🏷️ Release label (major vs. independent)
- 📅 Release timing (day of week, season)

<br/>

---

## 🔮 What's Next

- [ ] Scrape artist follower counts via Spotify API and retrain
- [ ] Add TikTok mention volume as a feature proxy
- [ ] Build a front-end UI for the prediction API
- [ ] Experiment with neural collaborative filtering
- [ ] Add SHAP explainability to the API response

<br/>

---

## 👤 Author

<div align="center">

**Hammad Malik**

Built with curiosity, caffeine, and a healthy respect for what the data actually says.

[![GitHub](https://img.shields.io/badge/GitHub-hammadmalik17-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hammadmalik17/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-hammad--malik---0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hammad-malik-)

</div>

<br/>

---

<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=1DB954&height=120&section=footer"/>

*"Well at least the insight was right."*

**⭐ Star this repo if you found it useful. PRs welcome.**

</div>
