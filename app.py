
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

st.set_page_config(
    page_title="Pakistan Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)


MODELS_DIR = "models"
DATA_DIR = "data"

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    features = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, "cleaned_car_data.csv"))

model, features = load_model()
df = load_data()


st.markdown("## 🚗 Pakistan Used Car Price Predictor")
st.markdown("Machine learning based **used car price estimation**")
st.markdown("Developed By Hamza Malik")


st.sidebar.header("🔧 Car Details")

make = st.sidebar.selectbox("Brand", sorted(df["make"].unique()))

model_name = st.sidebar.selectbox(
    "Model",
    sorted(df[df["make"] == make]["model"].unique())
)

current_year = datetime.now().year

year = st.sidebar.slider(
    "Manufacturing Year",
    int(df["year"].min()),
    current_year,
    2018
)

engine = st.sidebar.selectbox(
    "Engine Capacity (CC)",
    sorted(df["engine"].dropna().unique())
)

transmission = st.sidebar.selectbox(
    "Transmission",
    sorted(df["transmission"].unique())
)

fuel = st.sidebar.selectbox(
    "Fuel Type",
    sorted(df["fuel"].unique())
)

mileage = st.sidebar.slider(
    "Mileage (KM)",
    0,
    500000,
    50000,
    step=5000
)

city = st.sidebar.selectbox(
    "City",
    sorted(df["city"].unique())
)

assembly = st.sidebar.selectbox(
    "Assembly",
    sorted(df["assembly"].unique())
)

body = st.sidebar.selectbox(
    "Body Type",
    sorted(df["body"].unique())
)

# ----------------------------------------------------------------------------
# DERIVED FEATURES (MATCH TRAINING)
# ----------------------------------------------------------------------------
car_age = current_year - year
mileage_per_year = mileage / max(1, car_age)

if engine <= 1000:
    engine_category = "Small"
elif engine <= 1500:
    engine_category = "Medium"
elif engine <= 2000:
    engine_category = "Large"
elif engine <= 3000:
    engine_category = "V-Large"
else:
    engine_category = "X-Large"

if car_age <= 3:
    age_category = "New"
elif car_age <= 7:
    age_category = "Fairly Used"
elif car_age <= 15:
    age_category = "Used"
else:
    age_category = "Old"

luxury = ["Mercedes", "Bmw", "Audi", "Lexus", "Land Rover"]
economy = ["Suzuki", "Daihatsu", "Proton", "Faw"]

if make in luxury:
    brand_category = "Luxury"
elif make in economy:
    brand_category = "Economy"
else:
    brand_category = "Standard"

# ----------------------------------------------------------------------------
# INPUT DATAFRAME
# ----------------------------------------------------------------------------
input_data = {
    "make": make,
    "model": model_name,
    "year": year,
    "engine": engine,
    "transmission": transmission,
    "fuel": fuel,
    "mileage": mileage,
    "city": city,
    "assembly": assembly,
    "body": body,
    "car_age": car_age,
    "engine_category": engine_category,
    "age_category": age_category,
    "brand_category": brand_category
}

input_df = pd.DataFrame([input_data])
input_df = input_df[features["all_features"]]

# ----------------------------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------------------------
if st.sidebar.button("🔮 Predict Price"):
    with st.spinner("Predicting price..."):
        pred_log = model.predict(input_df)[0]
        price = np.expm1(pred_log)

        st.success(f"💰 Estimated Market Price: **PKR {price:,.0f}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Low Range", f"PKR {price*0.85:,.0f}")
        c2.metric("Expected", f"PKR {price:,.0f}")
        c3.metric("High Range", f"PKR {price*1.15:,.0f}")

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Overall Market", "📈 Brand & Model Trends"])

# ----------------------------------------------------------------------------
# TAB 1 - OVERALL MARKET
# ----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📊 Market Snapshot")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Price", f"PKR {df['price'].mean():,.0f}")
    c2.metric("Total Listings", f"{len(df):,}")
    c3.metric("Top Brand", df["make"].mode()[0])

    st.markdown("### 📈 Average Price by Year")
    year_avg = df.groupby("year")["price"].mean().reset_index()
    st.line_chart(year_avg.set_index("year"))

# ----------------------------------------------------------------------------
# TAB 2 - BRAND & MODEL TRENDS
# ----------------------------------------------------------------------------
with tab2:
    st.markdown("### 📈 Average Price by Year (Brand & Model)")

    col1, col2 = st.columns(2)

    selected_brand = col1.selectbox(
        "Select Brand",
        sorted(df["make"].unique()),
        key="brand_trend"
    )

    models_available = sorted(
        df[df["make"] == selected_brand]["model"].unique()
    )

    selected_model = col2.selectbox(
        "Select Model",
        models_available,
        key="model_trend"
    )

    trend_df = df[
        (df["make"] == selected_brand) &
        (df["model"] == selected_model)
    ]

    if len(trend_df) > 0:
        avg_price = (
            trend_df
            .groupby("year")["price"]
            .mean()
            .reset_index()
            .sort_values("year")
        )

        st.line_chart(avg_price.set_index("year"))

        st.caption(
            f"Showing average prices for **{selected_brand} {selected_model}**"
        )
    else:
        st.warning("No data available for this selection.")

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"<div style='text-align:center; color:gray;'>"
    f"© {datetime.now().year} Pakistan Car Price Predictor | Developed By Hamza Malik"
    f"</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center;'> Developed by <b>Hamza Malik</b></h3>",
    unsafe_allow_html=True
)

st.markdown("---")