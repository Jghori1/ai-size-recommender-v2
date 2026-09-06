import re
import numpy as np
import pandas as pd

# ============================================================================
# CUP SIZE MAPPING
# ============================================================================
CUP_MAP = {
    "aa": 0.5,
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "dd": 5,
    "ddd": 6,
    "e": 5,
    "f": 6,
    "g": 7,
    "h": 8,
    "i": 9,
    "j": 10,
}

# ============================================================================
# PARSER FUNCTIONS
# ============================================================================

def parse_height(h):
    """Convert '5\' 8\"' or '5 8' to inches (68.0)"""
    if pd.isna(h):
        return np.nan
    
    m = re.match(r"(\d+)'\s*(\d+)", str(h).strip())
    if not m:
        return np.nan
    
    feet = float(m.group(1))
    inches = float(m.group(2))
    return feet * 12 + inches


def parse_weight(w):
    """Convert '137lbs' or '137' to pounds (137.0)"""
    if pd.isna(w):
        return np.nan
    
    digits = re.sub(r"[^\d.]", "", str(w))
    
    if not digits:
        return np.nan
    
    return float(digits)


def parse_bust(b):
    """
    Convert '34d' to (band=34.0, cup=4.0)
    Handles edge cases: '34d+' -> (34.0, 4.5), '32ddd/e' -> (32.0, 5.5)
    """
    if pd.isna(b):
        return (np.nan, np.nan)
    
    b = str(b).strip().lower()
    
    # Extract band size (number at start)
    m = re.match(r"^(\d+)", b)
    if not m:
        return (np.nan, np.nan)
    band = float(m.group(1))
    
    # Extract cup letters (ignore +, /, digits, spaces)
    cup_str = re.sub(r"[^a-z]", "", b)
    
    if not cup_str:
        return (np.nan, np.nan)
    
    # Handle "/" (between two cup sizes)
    if "/" in b:
        parts = cup_str.split("/")
        
        # Only process if we actually have two parts
        if len(parts) == 2:
            cup1 = CUP_MAP.get(parts[0], np.nan)
            cup2 = CUP_MAP.get(parts[1], np.nan)
            
            # Average if both exist
            if not np.isnan(cup1) and not np.isnan(cup2):
                cup = (cup1 + cup2) / 2
            elif not np.isnan(cup1):
                cup = cup1
            else:
                cup = cup2
        else:
            # Fallback: just use the first part
            cup = CUP_MAP.get(parts[0], np.nan)
    else:
        # Normal case: just look up the letter
        cup = CUP_MAP.get(cup_str, np.nan)
    
    # Handle "+" (between sizes, add 0.5)
    if "+" in b and not np.isnan(cup):
        cup = cup + 0.5
    
    return (band, cup)


def parse_numeric(x):
    """Generic converter: try to extract a number"""
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


# ============================================================================
# MAIN CLEANING FUNCTION
# ============================================================================

def clean_dataframe(df):
    """
    Convert raw RentTheRunway data to clean numeric dataframe.
    
    Input: raw dataframe with string columns (height, weight, bust size, etc.)
    Output: clean dataframe with numeric columns, ready for modeling
    """
    out = pd.DataFrame(index=df.index)
    
    # --- IDENTIFIERS (keep as strings) ---
    out["user_id"] = df["user_id"].astype(str)
    out["item_id"] = df["item_id"].astype(str)
    
    # --- TARGET VARIABLE ---
    out["fit"] = df["fit"]
    
    # --- MEASUREMENTS (parse strings to numbers) ---
    out["height_in"] = df["height"].apply(parse_height)
    out["weight_lb"] = df["weight"].apply(parse_weight)
    out["age"] = df["age"].apply(parse_numeric)
    
    # Parse bust into two columns
    bust = df["bust size"].apply(parse_bust)
    out["bust_band"] = [b[0] for b in bust]
    out["bust_cup"] = [b[1] for b in bust]
    
    # --- SIZE (the size ordered) ---
    out["size"] = df["size"].apply(parse_numeric)
    
    # --- CATEGORIES (clean up strings) ---
    out["body_type"] = df["body type"].fillna("unknown").str.lower().str.strip()
    out["category"] = df["category"].fillna("unknown").str.lower().str.strip()
    out["rented_for"] = df["rented for"].fillna("unknown").str.lower().str.strip()
    
    # --- SANITY CHECKS (filter out impossible values) ---
    out.loc[~out["height_in"].between(48, 84), "height_in"] = np.nan
    out.loc[~out["weight_lb"].between(70, 400), "weight_lb"] = np.nan
    out.loc[~out["age"].between(12, 95), "age"] = np.nan
    
    # --- DROP UNUSABLE ROWS ---
    before = len(out)
    out = out.dropna(subset=["fit", "size"])
    dropped = before - len(out)
    print(f"Dropped {dropped} rows missing target/size")
    
    return out.reset_index(drop=True)
