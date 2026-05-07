from pydantic import BaseModel, Field

class SongFeatures(BaseModel):
    bpm: float              = Field(..., ge=0,   le=300, example=120,  description="Beats per minute")
    key: int                = Field(..., ge=0,   le=11,  example=5,    description="Musical key: C=0, C#=1 ... B=11")
    mode: int               = Field(..., ge=0,   le=1,   example=1,    description="1=Major, 0=Minor")
    danceability_pct: float = Field(..., ge=0,   le=100, example=75.0, description="Danceability %")
    valence_pct: float      = Field(..., ge=0,   le=100, example=60.0, description="Positivity/valence %")
    energy_pct: float       = Field(..., ge=0,   le=100, example=80.0, description="Energy %")
    acousticness_pct: float = Field(..., ge=0,   le=100, example=20.0, description="Acousticness %")
    speechiness_pct: float  = Field(..., ge=0,   le=100, example=5.0,  description="Speechiness %")
    liveness_pct: float     = Field(..., ge=0,   le=100, example=12.0, description="Liveness %")
    artist_count: int       = Field(..., ge=1,   le=20,  example=1,    description="Number of artists on the track")

    class Config:
        json_schema_extra = {
            "example": {
                "bpm": 120, "key": 5, "mode": 1,
                "danceability_pct": 75, "valence_pct": 60,
                "energy_pct": 80, "acousticness_pct": 20,
                "speechiness_pct": 5, "liveness_pct": 12,
                "artist_count": 1
            }
        }


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    predicted_streams: int
    log_streams: float
    model_used: str
    note: str
