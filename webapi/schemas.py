"""
Pydantic schemas for request / response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# ── Request ------------------------------------------------------------

class MediaItem(BaseModel):
    type: str = Field("thumbnail", description="'video' or 'thumbnail'")
    url: Optional[str] = None
    summary: Optional[str] = Field(
        None,
        description="AI-generated visual summary of the media. "
                    "If not provided, the system uses metadata only.",
    )


class ProfileStats(BaseModel):
    username: str = Field(..., description="Brand username, e.g. 'sprite_india'")
    followers: int = Field(0, ge=0)
    post_author_username: Optional[str] = None


class MetadataContent(BaseModel):
    caption: str = Field("", description="Post caption text")
    media_name: str = Field("reel", description="'reel', 'post', or 'album'")
    duration: int = Field(0, ge=0, description="Video duration in seconds (0 for images)")
    is_collaborated_post: bool = False
    collaborators: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None


class PredictRequest(BaseModel):
    metadata_content: MetadataContent
    profile_stats: ProfileStats
    media: List[MediaItem] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "metadata_content": {
                    "caption": "Garmi ke dar se dekhi horror film? Sprite Gatak! 🧊😎",
                    "media_name": "reel",
                    "duration": 30,
                    "is_collaborated_post": True,
                    "collaborators": ["some_creator"],
                    "created_at": "2026-04-20T18:30:00+00:00",
                },
                "profile_stats": {
                    "username": "sprite_india",
                    "followers": 170000,
                },
                "media": [
                    {
                        "type": "video",
                        "summary": "A woman drinking Sprite in summer heat.",
                    }
                ],
            }
        }


# ── Response -------------------------------------------------------──

class ExplanationItem(BaseModel):
    factor: str
    impact: str  # 'positive', 'negative', 'neutral'
    detail: str


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="'low', 'medium', or 'high' (Random Forest primary)")
    confidence: float
    primary_model: Optional[str] = Field(
        'random_forest',
        description="Which model produced the headline prediction.",
    )
    probabilities: Dict[str, float]
    model_predictions: Optional[Dict] = None
    shap_top_features: Optional[List[Dict]] = None
    explanation: List[ExplanationItem]
    brand_context: Dict
    features_used: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    brands_available: List[str]


class BrandStatsResponse(BaseModel):
    brand: str
    stats: Dict


class EvaluationResponse(BaseModel):
    baselines: Dict[str, float]
    cv_accuracy: float
    cv_f1_macro: float
    confusion_matrix: List[List[int]]
    per_brand: Dict


# ── Feedback -------------------------------------------------------──

class FeedbackRequest(BaseModel):
    prediction_id: str = Field("", description="ID of the prediction to correct (optional)")
    predicted_label: str = Field(..., description="The label that was predicted")
    correct_label: str = Field(..., description="The correct label (low/medium/high)")
    brand: str = Field("", description="Brand username")
    caption: str = Field("", description="Caption that was used")
    media_type: str = Field("reel", description="Content type")
    features: Dict[str, float] = Field(default_factory=dict,
                                        description="Features dict from prediction response")


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: Optional[str] = None
    quality_score: Optional[float] = None
    message: str


class DriftReportResponse(BaseModel):
    risk_level: Optional[str] = None
    drift_detected: Optional[bool] = None
    buffer_size: Optional[int] = None
    overall_psi: Optional[float] = None
    message: str
