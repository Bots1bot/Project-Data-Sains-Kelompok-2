import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide", page_icon="🏡")

@st.cache_resource
def load_model_file(path: str = "full_model.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, str(e)

model, load_error = load_model_file("full_model.pkl")

st.title("🏡 Prediksi Harga Rumah — Adaptif & Ramah Pengguna")

left, right = st.columns([2, 1])

with right:
    st.markdown("**Pengaturan model**")
    if model is not None:
        st.success("Model berhasil dimuat dari full_model.pkl")
    else:
        st.error("Model gagal dimuat dari full_model.pkl")
        st.info("Silakan unggah file model (.pkl) jika ingin menggunakan model lain")
        uploaded = st.file_uploader("Unggah file .pkl", type=["pkl"])
        if uploaded is not None:
            try:
                model = pickle.load(uploaded)
                st.success("Model berhasil dimuat dari unggahan Anda")
            except Exception as e:
                st.error(f"Gagal memuat model unggahan: {e}")

with left:
    st.header("Masukkan Data Properti")
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            bedrooms = st.slider("Kamar Tidur", 1, 8, 3)
            bathrooms = st.slider("Kamar Mandi", 1, 4, 2)
        with col2:
            land_size_m2 = st.number_input("Luas Tanah (m²)", min_value=1.0, max_value=10000.0, value=100.0, step=1.0, format="%.1f")
            building_size_m2 = st.number_input("Luas Bangunan (m²)", min_value=1.0, max_value=10000.0, value=90.0, step=1.0, format="%.1f")
        with col3:
            floors = st.slider("Jumlah Lantai", 1, 6, 2)
            city_choice = st.selectbox("Kota", [
                "Bekasi", "Bogor", "Depok", "Jakarta Barat", "Jakarta Pusat",
                "Jakarta Selatan", "Jakarta Timur", "Jakarta Utara", "Tangerang"
            ])
        furnishing_choice = st.selectbox("Furnishing", [
            "baru", "furnished", "semi furnished", "unfurnished"
        ])
        submit = st.form_submit_button("Prediksi Harga")

    sample_raw = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "land_size_m2": land_size_m2,
        "building_size_m2": building_size_m2,
        "floors": floors,
        "city": city_choice,
        "furnishing": furnishing_choice
    }])
    st.subheader("Ringkasan Input")
    st.table(sample_raw.T.rename(columns={0: "Nilai"}))

def build_onehot_dict(city, furnishing):
    cities = ["Bekasi", "Bogor", "Depok", "Jakarta Barat", "Jakarta Pusat",
              "Jakarta Selatan", "Jakarta Timur", "Jakarta Utara", "Tangerang"]
    furns = ["baru", "furnished", "semi furnished", "unfurnished"]
    d = {}
    for c in cities:
        d[f"city_ {c}"] = 1 if c == city else 0
        d[f"city_{c}"] = 1 if c == city else 0
    for f in furns:
        key_space = f"furnishing_{f}"
        key_normal = f"furnishing_{f}"
        d[key_space] = 1 if f == furnishing else 0
        d[key_normal] = 1 if f == furnishing else 0
    return d

def get_expected_features(model_obj):
    try:
        feats = getattr(model_obj, "feature_names_in_", None)
        return list(feats) if feats is not None else None
    except Exception:
        return None

def prepare_input_dataframe(raw_df, onehot_dict, expected_features):
    combined = {}
    combined.update(raw_df.iloc[0].to_dict())
    combined.update(onehot_dict)
    if expected_features is None:
        return pd.DataFrame([combined])
    row = {}
    for feat in expected_features:
        if feat in combined:
            row[feat] = combined[feat]
        else:
            row[feat] = 0
    return pd.DataFrame([row])

def safe_predict(model_obj, X):
    try:
        return model_obj.predict(X)
    except Exception:
        X_alt = X.reindex(sorted(X.columns), axis=1).fillna(0)
        return model_obj.predict(X_alt)

if submit:

    if building_size_m2 > land_size_m2:
        st.error("❌ Input tidak logis: Luas bangunan tidak boleh lebih besar daripada luas tanah. Silakan koreksi input terlebih dahulu.")
        st.stop()

    if model is None:
        st.error("Tidak ada model yang bisa dipakai untuk prediksi. Unggah model atau perbaiki file full_model.pkl.")
    else:
        expected_features = get_expected_features(model)
        onehot = build_onehot_dict(city_choice, furnishing_choice)
        X = prepare_input_dataframe(sample_raw, onehot, expected_features)
        try:
            with st.spinner("Melakukan prediksi..."):
                pred = safe_predict(model, X)
            val = float(np.array(pred).ravel()[0])
            st.subheader("Hasil Prediksi")
            currency = f"Rp {val:,.2f}"
            st.success(currency)
            st.info("Catatan: hasil prediksi bersifat estimasi berdasarkan model yang digunakan")
        except Exception as e:
            st.error(f"Prediksi gagal: {e}")

        if hasattr(model, "feature_names_in_"):
            try:
                st.write("Fitur yang digunakan model:")
                st.write(list(model.feature_names_in_))
            except Exception:
                pass

        if hasattr(model, "feature_importances_"):
            try:
                fi = np.array(model.feature_importances_)
                cols = X.columns.tolist()
                if len(fi) == len(cols):
                    imp_df = pd.DataFrame({
                        "feature": cols,
                        "importance": fi
                    }).sort_values("importance", ascending=False)
                    st.subheader("Peringkat Feature Importance")
                    st.table(imp_df.head(10).reset_index(drop=True))
            except Exception:
                pass

st.markdown("---")
st.caption("Aplikasi prediksi harga rumah ini bekerja menggunakan model machine learning. Semakin baik model dan data latih, semakin akurat hasil prediksi.")
