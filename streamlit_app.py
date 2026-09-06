import streamlit as st
import joblib
import pickle
import sys
import os

sys.path.insert(0, '.')

from src.inference import recommend_size

st.set_page_config(page_title="AI Size & Fit Recommender", layout="wide")

st.title("👗 Size & Fit Recommender")
st.markdown("### Find Your Perfect Clothing Size with AI")
st.write("Enter your measurements and item details below to get a personalized size recommendation.")

# Load model and data
@st.cache_resource
def load_models():
    model_path = "models/size_model_rf.joblib"
    if not os.path.exists(model_path):
        import gdown
        file_id = "1Gls9ZEfiqPSYQChcJmSeSRBf1J1KjvRu"
        url = f"https://drive.google.com/uc?id={file_id}"
        os.makedirs("models", exist_ok=True)
        gdown.download(url, model_path, quiet=False)
    
    MODEL = joblib.load(model_path)
    
    with open("data/processed/item_stats.pkl", "rb") as f:
        ITEM_STATS, CAT_STATS, GLOBAL_AVG = pickle.load(f)
    
    return MODEL, ITEM_STATS, CAT_STATS, GLOBAL_AVG

try:
    MODEL, ITEM_STATS, CAT_STATS, GLOBAL_AVG = load_models()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

FEATURE_COLS = [
    "height_in", "weight_lb", "bmi", "weight_per_inch",
    "bust_band", "bust_cup", "bust_total", "age",
    "size", "size_delta",
    "item_avg_fit_size", "item_fit_count", "item_txn_count",
    "item_size_std", "item_is_known",
    "body_type", "category", "rented_for",
]

st.markdown("---")
st.subheader("📝 Step 1: Your Body Measurements")
st.write("*These help us understand your body shape and size*")

col1, col2 = st.columns(2)

with col1:
    height = st.slider(
        "📏 Height (inches)",
        min_value=48,
        max_value=84,
        value=66,
        help="Your height in inches (e.g., 66 inches = 5'6\")"
    )
    bust_band = st.slider(
        "📐 Bust Band",
        min_value=28,
        max_value=52,
        value=34,
        help="Your natural bust measurement (band size)"
    )

with col2:
    weight = st.slider(
        "⚖️ Weight (lbs)",
        min_value=70,
        max_value=400,
        value=130,
        help="Your body weight in pounds"
    )
    bust_cup = st.select_slider(
        "🎀 Bust Cup Size",
        options=["AA", "A", "B", "C", "D", "DD", "DDD", "E", "F", "G", "H", "I", "J"],
        value="C",
        help="Your bust cup size (C, D, etc.)"
    )

st.markdown("---")
st.subheader("ℹ️ Step 2: Personal Information")
st.write("*Tell us about your style preferences and body type*")

col3, col4 = st.columns(2)

with col3:
    age = st.number_input(
        "🎂 Age",
        min_value=12,
        max_value=95,
        value=26,
        help="Your age (helps personalize recommendations)"
    )
    body_type = st.selectbox(
        "💃 Body Type",
        ["hourglass", "pear", "apple", "rectangle", "unknown"],
        help="Select the body shape that best describes you"
    )

with col4:
    rented_for = st.selectbox(
        "🎭 Occasion",
        ["party", "casual", "work", "wedding", "other"],
        help="What's the occasion for this clothing?"
    )

st.markdown("---")
st.subheader("👕 Step 3: Item Details")
st.write("*Tell us about the clothing item you're looking for*")

col5, col6 = st.columns(2)

with col5:
    item_id = st.text_input(
        "🏷️ Item ID (optional)",
        value="126335",
        help="The product ID (helps if you're renting from a specific place)"
    )

with col6:
    category = st.selectbox(
        "👗 Clothing Type",
        ["dress", "top", "bottom", "jacket", "intimate"],
        help="What type of clothing are you looking for?"
    )

st.markdown("---")

# Predict button
st.subheader("🔮 Get Your Recommendation")
if st.button("✨ Find My Perfect Size", use_container_width=True, key="predict_btn"):
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
        
        st.success("✅ Recommendation Ready!")
        st.markdown("---")
        
        st.subheader("📊 Your Results")
        
        col_rec, col_conf, col_known = st.columns(3)
        
        with col_rec:
            st.metric(
                "📏 Recommended Size",
                result['recommended_size']
            )
        
        with col_conf:
            confidence_pct = f"{result['confidence']:.1%}"
            st.metric(
                "🎯 Confidence Level",
                confidence_pct
            )
        
        with col_known:
            status = "✅ Yes" if result['item_known'] else "❌ No"
            st.metric(
                "📊 In Database",
                status
            )
        
        st.markdown("---")
        
        if result['alternatives']:
            st.subheader("🤔 Other Sizes to Consider")
            st.write("If the recommended size doesn't fit perfectly, try these alternatives:")
            
            for alt in result['alternatives'][:3]:
                st.write(f"• **Size {alt['size']}** — {alt['confidence']:.1%} match")
        
        st.markdown("---")
        st.info("�� **Tip:** Sizing can vary by brand and style. Always check the item's specific size guide if available!")
    
    except Exception as e:
        st.error(f"❌ Something went wrong: {str(e)}")
        st.write("Please check your inputs and try again.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ | Powered by Machine Learning</p>", unsafe_allow_html=True)
