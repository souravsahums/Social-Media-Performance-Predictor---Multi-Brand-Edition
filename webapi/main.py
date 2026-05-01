"""
FastAPI application – Social Media Performance Predictor API.
"""

import os
import sys
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from webapi.schemas import (
    PredictRequest, PredictResponse, HealthResponse,
    BrandStatsResponse, EvaluationResponse,
    FeedbackRequest, FeedbackResponse, DriftReportResponse,
)
from model.predictor import EngagementPredictor
from model.slm_predictor import SLMPredictor
from model.drift import DriftDetector
from model.feedback import FeedbackManager

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
WEBAPP_DIR = os.path.join(PROJECT_ROOT, 'webapp')
FEEDBACK_DIR = os.path.join(ARTIFACTS_DIR, 'feedback')

predictor: EngagementPredictor = None  # type: ignore
slm_predictor: SLMPredictor = None  # type: ignore
drift_detector: DriftDetector = None  # type: ignore
feedback_manager: FeedbackManager = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and supporting modules on startup."""
    global predictor, slm_predictor, drift_detector, feedback_manager
    if os.path.exists(os.path.join(ARTIFACTS_DIR, 'model.pt')):
        predictor = EngagementPredictor(ARTIFACTS_DIR)
        print("Model loaded successfully.")
    else:
        print("WARNING: No trained model found in artifacts/. "
              "Run `python -m model.train` first.")

    # SLM predictor (always available)
    brand_stats_path = os.path.join(ARTIFACTS_DIR, 'brand_stats.json')
    if os.path.exists(brand_stats_path):
        slm_predictor = SLMPredictor()
        slm_predictor.load_brand_stats(brand_stats_path)
        print("SLM predictor loaded.")

    # Drift detector
    drift_ref_path = os.path.join(ARTIFACTS_DIR, 'drift_reference.json')
    if os.path.exists(drift_ref_path):
        drift_detector = DriftDetector.load(drift_ref_path)
        print("Drift detector loaded.")

    # Feedback manager
    feedback_manager = FeedbackManager(FEEDBACK_DIR)
    print("Feedback manager initialized.")

    yield


app = FastAPI(
    title="Social Media Performance Predictor",
    description="Predict Instagram post engagement performance with explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------
# API Routes
# ----------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if predictor else "no_model",
        model_loaded=predictor is not None,
        brands_available=list(predictor.brand_stats.keys()) if predictor else [],
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(503, "Model not loaded. Train the model first.")

    # Build the data dict matching our internal format
    post_data = {
        'metadata_content': req.metadata_content.model_dump(),
        'profile_stats': req.profile_stats.model_dump(),
        'media': [m.model_dump() for m in req.media],
    }
    # Fill post_author_username if not provided
    if not post_data['profile_stats'].get('post_author_username'):
        post_data['profile_stats']['post_author_username'] = (
            post_data['profile_stats']['username']
        )

    # Extract visual features from thumbnail URL if media has URLs
    has_media_url = any(m.get('url') for m in post_data.get('media', []))

    try:
        result = predictor.predict(post_data, include_visual=has_media_url)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

    return PredictResponse(**result)


@app.post("/api/predict-simple")
async def predict_simple(
    caption: str = Form(""),
    brand: str = Form("sprite_india"),
    followers: int = Form(100000),
    media_type: str = Form("reel"),
    duration: int = Form(30),
    is_collab: bool = Form(False),
    collaborators: str = Form(""),
    visual_summary: str = Form(""),
    thumbnail_url: str = Form(""),
):
    """Simplified prediction endpoint for the demo frontend form."""
    if predictor is None:
        raise HTTPException(503, "Model not loaded. Train the model first.")

    collabs = [c.strip() for c in collaborators.split(',') if c.strip()]
    media = []
    if media_type == 'reel':
        media.append({'type': 'video', 'summary': visual_summary or None})
    if visual_summary or thumbnail_url:
        media.append({
            'type': 'thumbnail',
            'summary': visual_summary,
            'url': thumbnail_url or None,
        })

    post_data = {
        'metadata_content': {
            'caption': caption,
            'media_name': media_type,
            'duration': duration if media_type == 'reel' else 0,
            'is_collaborated_post': is_collab,
            'collaborators': collabs,
            'created_at': '',
        },
        'profile_stats': {
            'username': brand,
            'followers': followers,
            'post_author_username': brand,
        },
        'media': media,
    }

    # Use visual features if thumbnail URL provided
    include_visual = bool(thumbnail_url)

    try:
        result = predictor.predict(post_data, include_visual=include_visual)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

    # Drift monitoring: track prediction features
    if drift_detector is not None and 'features_used' in result:
        from model.features import FEATURE_COLUMNS
        import numpy as np
        fv = np.array([result['features_used'].get(c, 0) for c in FEATURE_COLUMNS],
                       dtype=np.float32)
        drift_detector.add_to_buffer(fv)
        drift_report = drift_detector.check_single(fv)
        result['drift_warning'] = drift_report.get('risk_level', 'unknown')

    # SLM second opinion
    if slm_predictor is not None:
        try:
            slm_result = slm_predictor.predict(post_data)
            result['slm_prediction'] = slm_result.get('prediction')
            result['slm_confidence'] = slm_result.get('confidence')
            result['slm_reasoning'] = slm_result.get('reasoning', [])
        except Exception:
            pass

    return result


# Dataset cache for URL lookups (lazy-loaded)
_dataset_cache: list = None


def _get_dataset():
    """Load and cache the assignment dataset for URL lookups."""
    global _dataset_cache
    if _dataset_cache is None:
        from data.loader import load_and_clean
        _dataset_cache = load_and_clean()
    return _dataset_cache


@app.post("/api/predict-url")
async def predict_from_url(url: str = Body(..., embed=True)):
    """Predict engagement for a post identified by its Instagram URL.

    Looks up the post in the training dataset by URL, extracts real features
    (including visual features from the S3 thumbnail), and runs the prediction
    pipeline. This allows evaluating real posts from the dataset.
    """
    if predictor is None:
        raise HTTPException(503, "Model not loaded. Train the model first.")

    if not url or not url.strip():
        raise HTTPException(422, "URL cannot be empty.")

    url = url.strip()

    # Normalize URL: strip trailing slashes to match stored URLs
    normalized_url = url.rstrip('/')

    # Search in dataset
    dataset = _get_dataset()
    matched_post = None
    for post in dataset:
        data = post.get('data', {})
        post_url = (data.get('url') or '').rstrip('/')
        if post_url and post_url == normalized_url:
            matched_post = data
            break

    if matched_post is None:
        # Try partial match (shortcode from URL)
        # Instagram URLs: https://www.instagram.com/p/SHORTCODE/ or /reel/SHORTCODE/
        import re
        match = re.search(r'/(p|reel|reels)/([A-Za-z0-9_-]+)', normalized_url)
        shortcode = match.group(2) if match else None
        if shortcode:
            for post in dataset:
                data = post.get('data', {})
                post_url = data.get('url') or ''
                if shortcode in post_url:
                    matched_post = data
                    break

    if matched_post is None:
        raise HTTPException(
            404,
            f"Post not found in dataset. Supported URLs are Instagram post/reel URLs "
            f"from our 5 brands (378 posts). Example: https://www.instagram.com/p/DUxE-r8AuL7/"
        )

    # Build post_data in the format the predictor expects
    post_data = {
        'metadata_content': matched_post.get('metadata_content', {}),
        'profile_stats': matched_post.get('profile_stats', {}),
        'media': matched_post.get('media', []),
    }

    # Find thumbnail URL from media list for visual features
    thumbnail_url = None
    for m in matched_post.get('media', []):
        if m.get('type') == 'thumbnail' and m.get('url'):
            thumbnail_url = m['url']
            break

    try:
        result = predictor.predict(post_data, include_visual=bool(thumbnail_url))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

    # Drift monitoring
    if drift_detector is not None and 'features_used' in result:
        from model.features import FEATURE_COLUMNS
        import numpy as np
        fv = np.array([result['features_used'].get(c, 0) for c in FEATURE_COLUMNS],
                       dtype=np.float32)
        drift_detector.add_to_buffer(fv)
        drift_report = drift_detector.check_single(fv)
        result['drift_warning'] = drift_report.get('risk_level', 'unknown')

    # SLM second opinion
    if slm_predictor is not None:
        try:
            slm_result = slm_predictor.predict(post_data)
            result['slm_prediction'] = slm_result.get('prediction')
            result['slm_confidence'] = slm_result.get('confidence')
            result['slm_reasoning'] = slm_result.get('reasoning', [])
        except Exception:
            pass

    # Add source metadata
    result['source_url'] = url
    result['source_brand'] = matched_post.get('profile_stats', {}).get('username', 'unknown')
    result['source_caption_preview'] = (
        matched_post.get('metadata_content', {}).get('caption', '')[:120]
    )

    return result


@app.get("/api/brands")
async def list_brands():
    if predictor is None:
        raise HTTPException(503, "Model not loaded.")
    return {
        'brands': [
            {'name': brand, 'stats': stats}
            for brand, stats in predictor.brand_stats.items()
        ]
    }


@app.get("/api/brand/{brand_name}", response_model=BrandStatsResponse)
async def brand_stats(brand_name: str):
    if predictor is None:
        raise HTTPException(503, "Model not loaded.")
    stats = predictor.brand_stats.get(brand_name)
    if stats is None:
        raise HTTPException(404, f"Brand '{brand_name}' not found.")
    return BrandStatsResponse(brand=brand_name, stats=stats)


@app.get("/api/evaluation")
async def evaluation_results():
    path = os.path.join(ARTIFACTS_DIR, 'evaluation_results.json')
    if not os.path.exists(path):
        raise HTTPException(404, "Evaluation not run yet. Run `python -m model.evaluate`.")
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------
# SLM / Reasoning Endpoint
# ----------------------------------------------

@app.post("/api/predict-slm")
async def predict_slm(
    caption: str = Form(""),
    brand: str = Form("sprite_india"),
    followers: int = Form(100000),
    media_type: str = Form("reel"),
    duration: int = Form(30),
    is_collab: bool = Form(False),
    collaborators: str = Form(""),
    visual_summary: str = Form(""),
):
    """Predict using the structured reasoning (SLM) predictor."""
    if slm_predictor is None:
        raise HTTPException(503, "SLM predictor not available.")

    collabs = [c.strip() for c in collaborators.split(',') if c.strip()]
    media = []
    if media_type == 'reel':
        media.append({'type': 'video', 'summary': visual_summary or None})
    if visual_summary:
        media.append({'type': 'thumbnail', 'summary': visual_summary})

    post_data = {
        'metadata_content': {
            'caption': caption, 'media_name': media_type,
            'duration': duration if media_type == 'reel' else 0,
            'is_collaborated_post': is_collab, 'collaborators': collabs,
            'created_at': '',
        },
        'profile_stats': {
            'username': brand, 'followers': followers,
            'post_author_username': brand,
        },
        'media': media,
    }
    try:
        result = slm_predictor.predict(post_data)
    except Exception as e:
        raise HTTPException(500, f"SLM prediction failed: {str(e)}")
    return result


# ----------------------------------------------
# Feedback Endpoints
# ----------------------------------------------

@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest):
    """Submit a correction when the prediction was wrong."""
    if feedback_manager is None:
        raise HTTPException(503, "Feedback system not available.")

    result = feedback_manager.submit(
        prediction_id=req.prediction_id,
        predicted_label=req.predicted_label,
        correct_label=req.correct_label,
        features=req.features,
        source_id='web_user',
        metadata={'brand': req.brand, 'caption': req.caption,
                  'media_type': req.media_type},
    )
    return FeedbackResponse(**result)


@app.get("/api/feedback/summary")
async def feedback_summary():
    """Get a summary of collected feedback."""
    if feedback_manager is None:
        raise HTTPException(503, "Feedback system not available.")
    return feedback_manager.get_summary()


@app.get("/api/feedback/retraining-pipeline")
async def retraining_pipeline():
    """Get the recommended retraining pipeline status."""
    if feedback_manager is None:
        raise HTTPException(503, "Feedback system not available.")
    return feedback_manager.get_retraining_pipeline()


# ----------------------------------------------
# Drift Monitoring Endpoints
# ----------------------------------------------

@app.get("/api/drift/status")
async def drift_status():
    """Get current drift monitoring status."""
    if drift_detector is None:
        return DriftReportResponse(
            message="Drift detector not initialized. Run training to generate reference data."
        )
    summary = drift_detector.get_drift_summary()
    return {
        'status': 'active',
        'buffer_size': summary['buffer_size'],
        'total_checks': summary['total_checks'],
        'drift_alerts': summary['drift_alerts'],
        'latest_report': summary.get('latest'),
        'message': 'Drift monitoring active.',
    }


@app.get("/api/drift/check")
async def drift_check():
    """Trigger a batch drift check on recent predictions."""
    if drift_detector is None:
        return DriftReportResponse(
            message="Drift detector not initialized."
        )
    report = drift_detector.check_batch_drift()
    if report is None:
        return DriftReportResponse(
            buffer_size=len(drift_detector._prediction_buffer),
            message="Not enough predictions in buffer (need 20+)."
        )
    return report


# ----------------------------------------------
# Serve frontend
# ----------------------------------------------

if os.path.exists(WEBAPP_DIR):
    app.mount("/static", StaticFiles(directory=WEBAPP_DIR), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(WEBAPP_DIR, 'index.html'))
