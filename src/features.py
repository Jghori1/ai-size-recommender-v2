import numpy as np
import pandas as pd

# ============================================================================
# BODY-LEVEL FEATURES (derived from user measurements)
# ============================================================================

def add_body_features(d):
    """
    Create derived features from raw measurements.
    
    Input: cleaned dataframe
    Output: same dataframe with new columns
    """
    d = d.copy()
    
    # BMI: captures body composition in one number
    # Formula: 703 * weight_lb / (height_in ^ 2)
    d["bmi"] = 703 * d["weight_lb"] / (d["height_in"] ** 2)
    
    # Bust total: rough proxy for overall bust circumference
    d["bust_total"] = d["bust_band"] + d["bust_cup"] * 1.0
    
    # Weight per inch: how compact/dense the build is
    d["weight_per_inch"] = d["weight_lb"] / d["height_in"]
    
    return d


# ============================================================================
# ITEM-LEVEL FEATURES (derived from all users' history with this item)
# ============================================================================
# ⚠️ IMPORTANT: These MUST be computed from TRAIN data only, 
# then applied to TEST data. Otherwise you leak test information.

def build_item_stats(train_df):
    """
    Compute item-level statistics from TRAINING rows only.
    
    Returns:
    - item_stats: DataFrame indexed by item_id with columns:
        * item_avg_fit_size: mean size where users rated "fit"
        * item_fit_count: how many times this item was rated "fit"
        * item_txn_count: total transactions for this item
        * item_size_std: std dev of sizes tried on this item
    - cat_stats: Series of category-level avg fit size (fallback for cold-start items)
    - global_avg: overall mean fit size (ultimate fallback)
    """
    
    # Filter to only "fit" outcomes
    fitting = train_df[train_df["fit"] == "fit"]
    
    # Per-item stats: what size typically fits?
    item_fit_stats = fitting.groupby("item_id")["size"].agg(
        item_avg_fit_size="mean",
        item_fit_count="count"
    )
    
    # Per-item stats: overall usage (all outcomes)
    item_all_stats = train_df.groupby("item_id")["size"].agg(
        item_txn_count="count",
        item_size_std="std"
    )
    
    # Merge them
    item_stats = item_fit_stats.join(item_all_stats, how="outer")
    
    # Per-category stats: fallback for items we've never seen
    cat_stats = fitting.groupby("category")["size"].mean().rename("cat_avg_fit_size")
    
    # Global stats: ultimate fallback
    global_avg = fitting["size"].mean()
    
    return item_stats, cat_stats, global_avg


def apply_item_stats(d, item_stats, cat_stats, global_avg):
    """
    Apply pre-computed item stats to a dataframe.
    Uses cascade: item stat → category stat → global mean
    
    Input: clean dataframe, and stats computed from train data
    Output: same dataframe with new item-level feature columns
    """
    d = d.copy()
    
    # Join item stats by item_id
    d = d.join(item_stats, on="item_id")
    
    # Join category stats by category
    d = d.join(cat_stats, on="category")
    
    # Fill missing values: cascade down the hierarchy
    d["cat_avg_fit_size"] = d["cat_avg_fit_size"].fillna(global_avg)
    d["item_avg_fit_size"] = d["item_avg_fit_size"].fillna(d["cat_avg_fit_size"])
    
    # Fill other missing item stats with 0 (means "no data")
    d["item_fit_count"] = d["item_fit_count"].fillna(0)
    d["item_txn_count"] = d["item_txn_count"].fillna(0)
    d["item_size_std"] = d["item_size_std"].fillna(0)
    
    # ★ THE MONEY FEATURE ★
    # How far is this candidate size from this item's typical fitting size?
    # If item_avg_fit_size=8 and we're trying size 10, delta=+2 (might be too loose)
    d["size_delta"] = d["size"] - d["item_avg_fit_size"]
    
    # Is this item well-understood (enough data), or are we guessing?
    d["item_is_known"] = (d["item_fit_count"] >= 3).astype(int)
    
    return d
