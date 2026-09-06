import streamlit as st
import joblib
import pickle
import sys
import os
from PIL import Image
from io import BytesIO
import requests

sys.path.insert(0, '.')

from src.inference import recommend_size

st.set_page_config(page_title="AI Size & Fit Recommender", layout="wide")

st.title("🎀 AI Size & Fit Recommender")
st.write("Get personalized clothing size recommendations based on your measurements")

# Load model and data
@st.cache_resource
def load_models():
    print("Loading models...")
    
    model_path = "models/size_model_rf.joblib"
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        import gdown
        file_id = "1Gls9ZEfiqPSYQChcJmSeSRBf1J1KjvRu"
        url = f"https://drive.google.com/uc?id={file_id}"
        os.makedirs("models", exist_ok=True)
        gdown.download(url, model_path, quiet=False)
    
    MODEL = joblib.load(model_path)
    print("✓ Model loaded")
    
    print("Loading item stats...")
    with open("data/processed/item_stats.pkl", "rb") as f:
        ITEM_STATS, CAT_STATS, GLOBAL_AVG = pickle.load(f)
    
    return MODEL, ITEM_STATS, CAT_STATS, GLOBAL_AVG

try:
    MODEL, ITEM_STATS, CAT_STATS, GLOBAL_AVG = load_models()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

FEATURE_COLS = [
    "height_in", "weight_lb", "bmi", "weight_per_inch",
    "bust_band", "bust_cup", "bust_total", "age",
    "size", "size_delta",
    "item_avg_fit_size", "item_fit_count", "item_txn_count",
    "item_size_std", "item_is_known",
    "body_type", "category", "rented_for",
]

# Fetch image from Pexels
def get_clothing_image(category):
    try:
        api_key = st.secrets.get("PEXELS_API_KEY")
        if not api_key:
            st.warning("API key not found")
            return None
        
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": api_key}
        params = {"query": f"{category} woman", "per_page": 1}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('photos'):
                img_url = data['photos'][0]['src']['medium']
                return img_url
        else:
            st.warning(f"Pexels API error: {response.status_code}")
    except Exception as e:
        st.warning(f"Image fetch error: {e}")
    
    return None

# User inputs
col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    st.subheader("👤 Your Measurements")
    height = st.slider("Height (inches)", 48, 84, 66)
    weight = st.slider("Weight (lbs)", 70, 400, 130)
    bust_band = st.slider("Bust Band", 28, 52, 34)
    bust_cup = st.select_slider("Bust Cup", ["AA", "A", "B", "C", "D", "DD", "DDD", "E", "F", "G", "H", "I", "J"], value="C")

with col2:
    st.subheader("ℹ️ Additional Info")
    age = st.number_input("Age", 12, 95, 26)
    body_type = st.selectbox("Body Type", ["hourglass", "pear", "apple", "rectangle", "unknown"])
    rented_for = st.selectbox("Rented For", ["party", "casual", "work", "wedding", "other"])

with col3:
    st.subheader("👗 Item Details")
    item_id = st.text_input("Item ID", "126335")
    category = st.selectbox("Category", ["dress", "top", "bottom", "jacket", "intimate"])
    
    # Display item image
    with st.spinner(f"Loading {category} image..."):
        img_url = get_clothing_image(category)
        if img_url:
            st.image(img_url, caption=f"{category.title()}", use_column_width=True)
        else:
            st.info(f"📸 {category.title()}")

# Predict button
st.markdown("---")
if st.button("🔮 Get Size Recommendation", use_container_width=True):
    try:
        cup_map = {"AA": 0.5, "A": 1, "B": 2, "C": 3, "D": 4, "DD": 5, "DDD": 6, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10}
        bust_cup_num = cup_map.get(bust_cup, 3)
        
        user_measurements = {
            "height_in": float(height),
            "weight_lb": float(weight),
            "bust_band": float(bust_band),
            "bust_cup": bust_cup_num,
            "age": int(age),
            "body_type": body_type,
            "rented_for": rented_for,
        }
        
        result = recommend_size(
            model=MODEL,
            user_measurements=user_measurements,
            item_id=item_id,
            item_category=category,
            candidate_sizes=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            item_stats=ITEM_STATS,
            cat_stats=CAT_STATS,
            global_avg=GLOBAL_AVG,
            feature_cols=FEATURE_COLS,
        )
        
        st.success("✅ Prediction Complete!")
        
        col_rec, col_conf, col_known = st.columns(3)
        
        with col_rec:
            st.metric("📏 Recommended Size", result['recommended_size'])
        
        with col_conf:
            st.metric("🎯 Confidence", f"{result['confidence']:.1%}")
        
        with col_known:
            st.metric("📊 Item Known?", "Yes" if result['item_known'] else "No")
        
        if result['alternatives']:
            st.subheader("📋 Alternative Sizes")
            for alt in result['alternatives'][:3]:
                st.write(f"• Size **{alt['size']}**: {alt['confidence']:.1%} confidence")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.markdown("<p style='text-align: center'>Made with ❤️ using Streamlit | ML Model: Random Forest Classifier</p>", unsafe_allow_html=True)
