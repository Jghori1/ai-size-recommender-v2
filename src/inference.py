"""
Inference functions for the Size & Fit Recommender.
Given user measurements and an item, predict the best size.
"""

import numpy as np
import pandas as pd
import joblib

def load_model(model_path="../models/size_model_rf.joblib"):
    """Load the trained Random Forest pipeline."""
    return joblib.load(model_path)

def recommend_size(
    model,
    user_measurements,
    item_id,
    item_category,
    candidate_sizes,
    item_stats,
    cat_stats,
    global_avg,
    feature_cols
):
    """
    Recommend the best size for a user, given measurements and an item.
    
    Args:
        model: trained scikit-learn pipeline
        user_measurements: dict with keys:
            - height_in, weight_lb, bust_band, bust_cup, age, body_type, rented_for
        item_id: string ID of the item
        item_category: category of the item (e.g., "dress", "gown")
        candidate_sizes: list of sizes to test (e.g., [0, 2, 4, 6, 8, 10, 12, 14, 16])
        item_stats, cat_stats, global_avg: precomputed from training data
        feature_cols: list of feature names (in order)
    
    Returns:
        dict with:
            - recommended_size: best size (as string)
            - confidence: probability of fit (0.0-1.0)
            - alternatives: list of runner-up sizes
            - item_known: whether we have training data for this item
    """
    
    # Build one row per candidate size
    rows = []
    for size in candidate_sizes:
        rows.append({
            "height_in": user_measurements["height_in"],
            "weight_lb": user_measurements["weight_lb"],
            "bust_band": user_measurements["bust_band"],
            "bust_cup": user_measurements["bust_cup"],
            "age": user_measurements.get("age", np.nan),
            "body_type": user_measurements.get("body_type", "unknown"),
            "rented_for": user_measurements.get("rented_for", "unknown"),
            "size": float(size),
            "category": item_category,
            "item_id": item_id,
        })
    
    df = pd.DataFrame(rows)
    
    # Add computed features
    df["bmi"] = 703 * df["weight_lb"] / (df["height_in"] ** 2)
    df["bust_total"] = df["bust_band"] + df["bust_cup"]
    df["weight_per_inch"] = df["weight_lb"] / df["height_in"]
    
    # Add item stats (from training data)
    df = df.join(item_stats, on="item_id")
    df = df.join(cat_stats, on="category")
    
    # Fill cascading: item → category → global
    df["cat_avg_fit_size"] = df["cat_avg_fit_size"].fillna(global_avg)
    df["item_avg_fit_size"] = df["item_avg_fit_size"].fillna(df["cat_avg_fit_size"])
    df["item_fit_count"] = df["item_fit_count"].fillna(0)
    df["item_txn_count"] = df["item_txn_count"].fillna(0)
    df["item_size_std"] = df["item_size_std"].fillna(0)
    df["size_delta"] = df["size"] - df["item_avg_fit_size"]
    df["item_is_known"] = (df["item_fit_count"] >= 3).astype(int)
    
    # Get probabilities for "fit" class
    X = df[feature_cols]
    probs = model.predict_proba(X)
    fit_idx = list(model.classes_).index("fit")
    p_fit = probs[:, fit_idx]
    
    # Rank by confidence
    ranked = sorted(zip(candidate_sizes, p_fit), key=lambda x: -x[1])
    best_size, best_confidence = ranked[0]
    
    # Alternatives (next 3)
    alternatives = [
        {"size": str(s), "confidence": round(float(p), 3)}
        for s, p in ranked[1:4]
    ]
    
    return {
        "recommended_size": str(int(best_size)),
        "confidence": round(float(best_confidence), 3),
        "is_confident": bool(best_confidence >= 0.50),
        "alternatives": alternatives,
        "item_known": bool(df["item_is_known"].iloc[0] == 1),
        "model_version": "v1.0.0",
    }
