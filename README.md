# 🎵 Spotify Hit Prediction — End-to-End ML Pipeline

> Can audio features predict a song's popularity? This project builds a complete ML pipeline to find out — from raw data to a live prediction API.

---

## 📌 Project Overview

This project builds an end-to-end machine learning system that predicts the number of streams a song will receive on Spotify, based purely on its **audio characteristics** — things like BPM, danceability, energy, and valence.

The core finding is itself a data science insight: **audio features alone explain only ~7% of stream variance**. Artist reach, editorial playlists, and social signals dominate — but those weren't in our dataset. Documenting this honestly is part of good ML practice.

---

## 🗂️ Project Structure

```
spotify-hit-prediction/
│
├── spotify-2023.csv           # Raw dataset (953 songs, Kaggle)
├── clean.ipynb                # Full notebook: EDA → Feature Eng → Modelling
│
├── model/
│   └── hit_predictor.pkl      # Saved best model (Gradient Boosting)
│
├── app/
│   ├── main.py                # FastAPI prediction service
│   ├── schema.py              # Request/response data models
│   └── requirements.txt       # API dependencies
│
├── Dockerfile                 # Containerised API
├── docker-compose.yml         # One-command run
│
├── mlruns/                    # MLflow experiment logs (auto-generated)
│
└── README.md
```

---

## 🔬 ML Pipeline

### 1. Data Preprocessing
- Loaded 953 Spotify songs from the [Kaggle Most Streamed Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023) dataset
- Handled encoding issues, removed commas from numeric strings, dropped nulls
- Dropped post-release leakage features (playlist counts, chart positions)

### 2. Feature Engineering
- **Yeo-Johnson transform** on skewed features (BPM, artist count, acousticness, etc.)
- **Log-transform** on target variable (`streams`) to reduce right skew
- Encoded `key` (C → 0, C# → 1 ... B → 11) and `mode` (Major → 1, Minor → 0)

### 3. Models Trained

| Model | R² | RMSE | CV R² |
|---|---|---|---|
| Linear Regression | 0.0326 | 1.4233 | -0.198 |
| Ridge Regression | 0.0326 | 1.4233 | -0.198 |
| Random Forest | 0.0685 | 1.3967 | -0.253 |
| **Gradient Boosting** ✓ | **0.0708** | **1.3950** | -0.348 |

> **Best model:** Gradient Boosting — saved to `model/hit_predictor.pkl`

### 4. Top Features (by Random Forest importance)

```
bpm                  0.150
valence_%            0.134
acousticness_%_yj    0.129
energy_%             0.191
danceability_%       0.187
```

### 5. Experiment Tracking
All runs logged with **MLflow** — params, metrics, and model artifacts versioned under experiment `spotify-hit-prediction`.

```bash
mlflow ui   # View at http://localhost:5000
```

---

## 🚀 Running the API

### Option 1 — Local (Python)

```bash
pip install -r app/requirements.txt
uvicorn app.main:app --reload
```

### Option 2 — Docker

```bash
docker build -t spotify-predictor .
docker run -p 8000:8000 spotify-predictor
```

### Option 3 — Docker Compose

```bash
docker-compose up
```

API is live at: `http://localhost:8000`  
Swagger docs at: `http://localhost:8000/docs`

---

## 📡 API Usage

### `POST /predict`

**Request body:**
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
  "predicted_streams": 284592103,
  "log_streams": 19.47,
  "model_used": "GradientBoostingRegressor",
  "note": "Audio features explain ~7% of stream variance. Treat as a rough signal."
}
```

### `GET /health`
```json
{ "status": "ok", "model": "GradientBoostingRegressor" }
```

---

## 🧰 Tech Stack

| Layer | Tool |
|---|---|
| Data processing | pandas, numpy, scipy |
| ML models | scikit-learn |
| Experiment tracking | MLflow |
| API | FastAPI + Uvicorn |
| Containerisation | Docker |
| Environment | Python 3.10 |

---

## 📊 Key Insight

> Audio features alone are **weak predictors** of a song's stream count (R² ≈ 0.07). This is a real-world finding: what a song *sounds like* is far less predictive than *who made it* and *how it was promoted*. A negative cross-validated R² confirms the model doesn't generalise well — and that's the honest result.

What would improve this model:
- Artist follower count at time of release
- Spotify editorial playlist inclusion flag
- Social media momentum (TikTok, Instagram mentions)

---

## 👤 Author

**Your Name**  
[GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📄 Dataset

Kaggle: [Top Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023) by Nidula Elgiriyewithana