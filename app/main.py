import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schema import SongFeatures, PredictionResponse

# ── Load model + transformer once at startup ──────────────────────────────────
try:
    model_package = joblib.load("model/hit_predictor.pkl")
    MODEL        = model_package["model"]
    FEATURE_COLS = model_package["feature_cols"]
    TRANSFORMER  = model_package["transformer"]
    SKEWED_COLS  = model_package["skewed_cols"]
    MODEL_NAME   = type(MODEL).__name__
    print(f"Model loaded: {MODEL_NAME}")
    print(f"Features    : {FEATURE_COLS}")
    print(f"Skewed cols : {SKEWED_COLS}")
except FileNotFoundError:
    raise RuntimeError("model/hit_predictor.pkl not found. Run the notebook first.")
except KeyError as e:
    raise RuntimeError(
        f"model_package missing key: {e}. "
        "Re-run Cell 24 in the notebook to re-save with transformer included."
    )


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "🎵 Spotify Hit Predictor",
    description = (
        "Predicts the number of Spotify streams for a song based on its audio features. "
        "Built on a Gradient Boosting model trained on the Top Spotify Songs 2023 dataset. "
        "Note: audio features explain ~7% of stream variance — treat predictions as rough signals."
    ),
    version = "1.0.0"
)


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(song: SongFeatures) -> pd.DataFrame:
    """
    Mirrors the exact transformations applied during training:
    1. Map API fields → original column names
    2. Apply the fitted PowerTransformer (same lambda as training)
    3. Return columns in the exact order the model expects
    """
    # Step 1: build a dataframe with original column names
    raw = pd.DataFrame([{
        'artist_count'  : float(song.artist_count),
        'bpm'           : float(song.bpm),
        'key'           : float(song.key),
        'mode'          : float(song.mode),
        'danceability_%': float(song.danceability_pct),
        'valence_%'     : float(song.valence_pct),
        'energy_%'      : float(song.energy_pct),
        'acousticness_%': float(song.acousticness_pct),
        'speechiness_%' : float(song.speechiness_pct),
        'liveness_%'    : float(song.liveness_pct),
    }])

    # Step 2: apply the SAME fitted PowerTransformer from training
    # This uses the learned lambda — not a fresh fit
    transformed = TRANSFORMER.transform(raw[SKEWED_COLS])
    for i, col in enumerate(SKEWED_COLS):
        raw[f'{col}_yj'] = transformed[:, i]

    # Step 3: return only the model's expected features in the right order
    return raw[FEATURE_COLS]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Check if the API and model are loaded correctly."""
    return {
        "status"  : "ok",
        "model"   : MODEL_NAME,
        "features": FEATURE_COLS
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(song: SongFeatures):
    """
    Predict the number of Spotify streams for a song
    given its audio characteristics.
    """
    try:
        # Engineer features exactly as during training
        features = engineer_features(song)

        # Predict log_streams, then reverse the log transform
        log_pred    = MODEL.predict(features)[0]
        stream_pred = int(np.expm1(log_pred))

        return PredictionResponse(
            predicted_streams = max(stream_pred, 0),   # clamp negatives to 0
            log_streams       = round(float(log_pred), 4),
            model_used        = MODEL_NAME,
            note              = (
                "Audio features explain ~7% of stream variance. "
                "This is a rough signal — not a chart guarantee."
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))