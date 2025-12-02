# app.py (adaptif)
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide")

# ---------------------------
# Load model
# ---------------------------
try:
    with open("full_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Try to obtain expected feature names if available
expected_features = None
try:
    expected_features = list(model.feature_names_in_)
except Exception:
    # not all sklearn estimators expose feature_names_in_
    expected_features = None

st.title("🏡 Prediksi Harga Rumah (Adaptif Input)")

# ---------------------------
# Sidebar inputs (user)
# ---------------------------
st.sidebar.header("Input Properti Rumah")
bedrooms = st.sidebar.slider("Jumlah Kamar Tidur", 1, 8, 3)
bathrooms = st.sidebar.slider("Jumlah Kamar Mandi", 1, 4, 2)
land_size_m2 = st.sidebar.slider("Luas Tanah (m²)", 10.0, 400.0, 100.0)
building_size_m2 = st.sidebar.slider("Luas Bangunan (m²)", 10.0, 400.0, 90.0)
floors = st.sidebar.slider("Jumlah Lantai", 1, 3, 2)

city_choice = st.sidebar.selectbox("Kota", [
    "Bekasi", "Bogor", "Depok", "Jakarta Barat", "Jakarta Pusat",
    "Jakarta Selatan", "Jakarta Timur", "Jakarta Utara", "Tangerang"
])

furnishing_choice = st.sidebar.selectbox("Furnishing", [
    "baru", "furnished", "semi furnished", "unfurnished"
])

# Known one-hot column lists (sesuaikan jika berbeda)
onehot_cities = [
    'city_ Bekasi','city_ Bogor','city_ Depok','city_ Jakarta Barat',
    'city_ Jakarta Pusat','city_ Jakarta Selatan','city_ Jakarta Timur',
    'city_ Jakarta Utara','city_ Tangerang'
]
onehot_furnish = [
    'furnishing_baru','furnishing_furnished',
    'furnishing_semi furnished','furnishing_unfurnished'
]

# ---------------------------
# Build candidate inputs
# ---------------------------
# raw (non-encoded) version
raw_df = pd.DataFrame([{
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'land_size_m2': land_size_m2,
    'building_size_m2': building_size_m2,
    'floors': floors,
    'city': city_choice,
    'furnishing': furnishing_choice
}])

# one-hot encoded version (as model might expect)
oh = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'land_size_m2': land_size_m2,
    'building_size_m2': building_size_m2,
    'floors': floors
}
for c in onehot_cities:
    oh[c] = 1 if c == f"city_ {city_choice}" else 0
for f in onehot_furnish:
    oh[f] = 1 if f == f"furnishing_{furnishing_choice}" else 0
oh_df = pd.DataFrame([oh])

# Show user's input summary
st.subheader("Input pengguna (ringkasan):")
st.write(raw_df)

# ---------------------------
# Decide which input to send to model
# ---------------------------
def predict_with_appropriate_input():
    # If model exposes expected feature names, use them to choose
    if expected_features is not None:
        exp_set = set(expected_features)
        # if model expects 'city' and 'furnishing' -> use raw_df
        if {'city', 'furnishing'}.issubset(exp_set):
            # check raw_df has the columns model wants
            missing = list(exp_set - set(raw_df.columns))
            if missing:
                st.error(f"Model expects columns {missing} but raw input lacks them.")
                return None
            X = raw_df[expected_features] if set(expected_features).issubset(set(raw_df.columns)) else raw_df.reindex(columns=expected_features).fillna(0)
            return model.predict(X)
        # if model expects one-hot columns -> use oh_df
        elif exp_set.issuperset(set(onehot_cities + onehot_furnish)):
            missing = list(exp_set - set(oh_df.columns))
            if missing:
                st.error(f"Model expects one-hot columns {missing} but constructed input lacks them.")
                return None
            X = oh_df[expected_features]
            return model.predict(X)
        else:
            # fallback: try raw, then one-hot
            try:
                return model.predict(raw_df)
            except Exception:
                try:
                    return model.predict(oh_df)
                except Exception as e:
                    st.error(f"Kedua percobaan prediksi (raw & one-hot) gagal: {e}")
                    return None
    else:
        # model does not expose feature names -> try raw first, then one-hot
        try:
            return model.predict(raw_df)
        except Exception:
            try:
                return model.predict(oh_df)
            except Exception as e:
                st.error(f"Kedua percobaan prediksi (raw & one-hot) gagal: {e}")
                return None

# ---------------------------
# Predict button
# ---------------------------
if st.sidebar.button("Prediksi Harga"):
    pred = predict_with_appropriate_input()
    if pred is not None:
        try:
            val = float(np.array(pred).ravel()[0])
            st.subheader("Hasil Prediksi:")
            st.success(f"Rp {val:,.2f}")
        except Exception as e:
            st.error(f"Gagal mengolah output prediksi: {e}")
