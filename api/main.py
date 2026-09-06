"""
AI Size & Fit Recommender — FastAPI Backend
"""

import sys
import os
import pickle
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
from src.inference import recommend_size

app = FastAPI(
    title="AI Size & Fit Recommender",
    description="Predicts clothing size based on user measurements",
    version="1.0.0",
)

MODEL = None
ITEM_STATS = None
CAT_STATS = None
GLOBAL_AVG = None
FEATURE_COLS = None

class SizeRecommendationRequest(BaseModel):
    height_in: float = Field(..., ge=48, le=84)
    weight_lb: float = Field(..., ge=70, le=400)
    bust_band: float = Field(..., ge=28, le=52)
    bust_cup: int = Field(..., ge=0, le=10)
    age: Optional[int] = Field(None, ge=12, le=95)
    body_type: str = Field("unknown")
    rented_for: str = Field("other")
    item_id: str = Field(...)
    category: str = Field("dress")
    candidate_sizes: List[int] = Field(default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

class SizeAlternative(BaseModel):
    size: str
    confidence: float

class SizeRecommendationResponse(BaseModel):
    recommended_size: str
    confidence: float
    is_confident: bool
    alternatives: List[SizeAlternative]
    item_known: bool
    model_version: str

@app.on_event("startup")
async def load_model():
    global MODEL, ITEM_STATS, CAT_STATS, GLOBAL_AVG, FEATURE_COLS
    
    try:
        print("Loading model...")
        
        # Check if model exists locally
        if os.path.exists("models/size_model_rf.joblib"):
            print("Loading local model...")
            MODEL = joblib.load("models/size_model_rf.joblib")
            print("✓ Local model loaded")
        else:
            print("Local model not found, downloading from Google Drive...")
            try:
                import gdown
                file_id = "1Gls9ZEfiqPSYQChcJmSeSRBf1J1KjvRu"
                url = f"https://drive.google.com/uc?id={file_id}"
                os.makedirs("models", exist_ok=True)
                gdown.download(url, "models/size_model_rf.joblib", quiet=False)
                MODEL = joblib.load("models/size_model_rf.joblib")
                print("✓ Model downloaded and loaded")
            except Exception as e:
                print(f"Failed to download model: {e}")
                print("Continuing without model...")
        
        print("Loading item stats...")
        with open("data/processed/item_stats.pkl", "rb") as f:
            ITEM_STATS, CAT_STATS, GLOBAL_AVG = pickle.load(f)
        print("✓ Item stats loaded")
        
        FEATURE_COLS = [
            "height_in", "weight_lb", "bmi", "weight_per_inch",
            "bust_band", "bust_cup", "bust_total", "age",
            "size", "size_delta",
            "item_avg_fit_size", "item_fit_count", "item_txn_count",
            "item_size_std", "item_is_known",
            "body_type", "category", "rented_for",
        ]
        
        print("✓ All models loaded successfully")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        import traceback
        traceback.print_exc()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "version": "1.0.0",
    }

@app.post("/predict-size", response_model=SizeRecommendationResponse)
async def predict_size(req: SizeRecommendationRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        user_measurements = {
            "height_in": req.height_in,
            "weight_lb": req.weight_lb,
            "bust_band": req.bust_band,
            "bust_cup": req.bust_cup,
            "age": req.age,
            "body_type": req.body_type,
            "rented_for": req.rented_for,
        }
        
        result = recommend_size(
            model=MODEL,
            user_measurements=user_measurements,
            item_id=req.item_id,
            item_category=req.category,
            candidate_sizes=req.candidate_sizes,
            item_stats=ITEM_STATS,
            cat_stats=CAT_STATS,
            global_avg=GLOBAL_AVG,
            feature_cols=FEATURE_COLS,
        )
        
        return SizeRecommendationResponse(
            recommended_size=result["recommended_size"],
            confidence=result["confidence"],
            is_confident=result["is_confident"],
            alternatives=[
                SizeAlternative(size=alt["size"], confidence=alt["confidence"])
                for alt in result["alternatives"]
            ],
            item_known=result["item_known"],
            model_version=result["model_version"],
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "AI Size & Fit Recommender API",
        "docs": "/docs",
        "health": "/health",
    }
