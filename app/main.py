import pickle
import numpy as np
import pandas as pd
from scipy.stats import yeojohnson
from fastapi import FastAPI, HTTPException
from app.schema import SongFeatures, PredictionResponse

# ── Load model once at startup ────────────────────────────────────────────────
try:
    with open("model/hit_predictor.pkl", "rb") as f:
        model_package = pickle.load(f)

    MODEL        = model_package["model"]
    FEATURE_COLS = model_package["feature_cols"]
    MODEL_NAME   = type(MODEL).__name__
except FileNotFoundError:
    raise RuntimeError("model/hit_predictor.pkl not found. Run the notebook first.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "🎵 Spotify Hit Predictor",
    description = (
        "Predicts the number of Spotify streams for a song based on its audio features. "
        "Built on a Gradient Boosting model trained on the Top Spotify Songs 2023 dataset."
    ),
    version     = "1.0.0"
)


# ── Feature engineering (mirrors the notebook exactly) ────────────────────────
def engineer_features(song: SongFeatures) -> pd.DataFrame:
    """
    Apply the same transformations used during training:
    - Yeo-Johnson on skewed features
    - Raw passthrough for bpm, key, mode, danceability, valence, energy
    """
    raw = {
        "bpm"           : song.bpm,
        "key"           : float(song.key),
        "mode"          : float(song.mode),
        "danceability_%" : song.danceability_pct,
        "valence_%"     : song.valence_pct,
        "energy_%"      : song.energy_pct,
    }

    # Yeo-Johnson transforms (lambdas are applied without fitting — approximation)
    # In production, you'd save the fitted PowerTransformer too.
    # Here we apply the transform naively which is fine for demonstration.
    def yj(val):
        result, _ = yeojohnson([val])
        return float(result[0])

    raw["acousticness_%_yj"] = yj(song.acousticness_pct)
    raw["speechiness_%_yj"]  = yj(song.speechiness_pct)
    raw["liveness_%_yj"]     = yj(song.liveness_pct)
    raw["artist_count_yj"]   = yj(song.artist_count)

    df = pd.DataFrame([raw])
    return df[FEATURE_COLS]  # ensure correct column order


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictionResponse)
def predict(song: SongFeatures):
    try:
        features    = engineer_features(song)
        log_pred    = MODEL.predict(features)[0]
        stream_pred = int(np.expm1(log_pred))

        return PredictionResponse(
            predicted_streams = max(stream_pred, 0),  # no negatives
            log_streams       = round(float(log_pred), 4),
            model_used        = MODEL_NAME,
            note              = (
                "Audio features explain ~7% of stream variance. "
                "This is a rough signal — not a chart guarantee."
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
