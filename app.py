"""
Streamlit web app: Used Car Selling Price Predictor.

Run locally:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

from utils.preprocessing import FEATURE_COLUMNS

MODEL_PATH = "model/car_price_model.joblib"
METADATA_PATH = "model/metadata.joblib"

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


model, metadata = load_artifacts()

st.title("🚗 Used Car Price Predictor")
st.write(
    "Estimate the fair resale price of a used car based on its specs. "
    "The model is a Random Forest Regressor trained on ~7,000 real used-car listings."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This app predicts used-car selling price (₹) from listing details "
        "such as brand, year, kilometers driven, fuel type, and engine specs."
    )
    m = metadata["metrics"]
    st.metric("R² score", f"{m['r2']:.3f}")
    st.metric("Mean Absolute Error", f"₹ {m['mae']:,.0f}")
    st.caption(f"Trained on {metadata['n_rows_trained']:,} listings.")

st.subheader("Enter car details")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", metadata["brands"])
    year = st.slider(
        "Year of manufacture",
        min_value=metadata["year_min"],
        max_value=metadata["year_max"] + 1,
        value=min(2018, metadata["year_max"]),
    )
    km_driven = st.number_input(
        "Kilometers driven",
        min_value=0,
        max_value=1_000_000,
        value=int(metadata["km_driven_median"]),
        step=1000,
    )
    fuel = st.selectbox("Fuel type", metadata["fuels"])
    seller_type = st.selectbox("Seller type", metadata["seller_types"])
    transmission = st.selectbox("Transmission", metadata["transmissions"])

with col2:
    owner = st.selectbox("Ownership", metadata["owners"])
    mileage = st.number_input(
        "Mileage (kmpl)",
        min_value=0.0,
        max_value=50.0,
        value=round(metadata["mileage_median"], 1),
        step=0.1,
    )
    engine = st.number_input(
        "Engine capacity (CC)",
        min_value=500,
        max_value=6000,
        value=int(metadata["engine_median"]),
        step=50,
    )
    max_power = st.number_input(
        "Max power (bhp)",
        min_value=20.0,
        max_value=500.0,
        value=round(metadata["max_power_median"], 1),
        step=1.0,
    )
    torque = st.number_input(
        "Torque (Nm)",
        min_value=20.0,
        max_value=1000.0,
        value=round(metadata["torque_median"], 1),
        step=5.0,
    )
    seats = st.selectbox(
        "Seats",
        [2, 4, 5, 6, 7, 8, 9, 10, 14],
        index=[2, 4, 5, 6, 7, 8, 9, 10, 14].index(int(metadata["seats_mode"]))
        if int(metadata["seats_mode"]) in [2, 4, 5, 6, 7, 8, 9, 10, 14]
        else 2,
    )

st.divider()

if st.button("Predict selling price", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "brand": brand,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "torque": torque,
        "seats": seats,
    }])[FEATURE_COLUMNS]

    prediction = model.predict(input_df)[0]

    st.success(f"### Estimated selling price: ₹ {prediction:,.0f}")
    st.caption(
        "This is an estimate from a machine learning model trained on historical "
        "listings. Actual market prices may vary."
    )

st.divider()
st.caption("Built with scikit-learn + Streamlit. Dataset: Car_details_v3 (used car listings).")
