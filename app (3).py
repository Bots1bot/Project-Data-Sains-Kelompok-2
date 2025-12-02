import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide", page_icon="🏡")

@st.cache_resource
def load_model_file(path: str = "full_model.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, str(e)

model, load_error = load_model_file()

st.markdown("<h1 style='text-align: center; color:#003566;'>🏡 Prediksi Harga Rumah</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:17px;'>Masukkan detail properti untuk mendapatkan estimasi harga terbaik</p>", unsafe_allow_html=True)
st.markdown("---")

if model:
    st.success("Model berhasil dimuat ✓")
else:
    st.error("Model gagal dimuat dari full_model.pkl")
    uploaded = st.file_uploader("🔄 Unggah file model (.pkl)", type=["pkl"])
    if uploaded:
        try:
            model = pickle.load(uploaded)
            st.success("Model berhasil dimuat dari unggahan ✓")
        except Exception as e:
            st.error(f"Gagal memuat file: {e}")

st.subheader("📝 Masukkan Data Properti")
with st.form("input_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        bedrooms = st.slider("🛏 Jumlah Kamar Tidur", 1, 8, 3)
        bathrooms = st.slider("🚿 Jumlah Kamar Mandi", 1, 4, 2)
    with col2:
        land_size_m2 = st.number_input("🌿 Luas Tanah (m²)", min_value=1.0, max_value=20000.0, value=100.0, step=1.0)
        building_size_m2 = st.number_input("🏠 Luas Bangunan (m²)", min_value=1.0, max_value=20000.0, value=90.0, step=1.0)
    with col3:
        floors = st.slider("🏢 Jumlah Lantai", 1, 6, 2)
        city_choice = st.selectbox("📍 Kota", [
            "Bekasi", "Bogor", "Depok", "Jakarta Barat", "Jakarta Pusat",
            "Jakarta Selatan", "Jakarta Timur", "Jakarta Utara", "Tangerang"
        ])
    furnishing_choice = st.selectbox("🛋 Furnishing", [
        "baru", "furnished", "semi furnished", "unfurnished"
    ])
    submit = st.form_submit_button("🔍 Prediksi Harga Rumah")

sample_raw = pd.DataFrame([{
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "land_size_m2": land_size_m2,
    "building_size_m2": building_size_m2,
    "floors": floors,
    "city": city_choice,
    "furnishing": furnishing_choice
}])

def build_onehot(city, furnishing):
    cities = ["Bekasi","Bogor","Depok","Jakarta Barat","Jakarta Pusat",
              "Jakarta Selatan","Jakarta Timur","Jakarta Utara","Tangerang"]
    furns = ["baru","furnished","semi furnished","unfurnished"]
    d = {}
    for c in cities:
        d[f"city_ {c}"] = 1 if c == city else 0
    for f in furns:
        d[f"furnishing_{f}"] = 1 if f == furnishing else 0
    return d

def get_expected_features(model_obj):
    return list(getattr(model_obj, "feature_names_in_", []))

def prepare_input(raw_df, onehot_dict, expected):
    merged = {**raw_df.iloc[0].to_dict(), **onehot_dict}
    return pd.DataFrame([{c: merged.get(c, 0) for c in expected}]) if expected else pd.DataFrame([merged])

def safe_predict(model_obj, X):
    try:
        return model_obj.predict(X)
    except:
        X_sorted = X.reindex(sorted(X.columns), axis=1).fillna(0)
        return model_obj.predict(X_sorted)

if submit:
    if building_size_m2 > land_size_m2:
        st.error("⚠ Luas bangunan tidak boleh lebih besar dari luas tanah.")
        st.stop()
    if model is None:
        st.error("Model belum siap digunakan.")
        st.stop()
    onehot = build_onehot(city_choice, furnishing_choice)
    expected_features = get_expected_features(model)
    X = prepare_input(sample_raw, onehot, expected_features)
    with st.spinner("⏳ Menghitung prediksi..."):
        pred = safe_predict(model, X)
        price = float(np.array(pred).ravel()[0])

    st.markdown("""
        <style>
            .result-box {
                background: #e9f5ff;
                padding: 25px;
                border-radius: 12px;
                border-left: 10px solid #0077b6;
            }
            .result-text { font-size: 30px; font-weight: bold; color: #004a7c; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="result-box">
            <div class="result-text">💰 Estimasi Harga: Rp {price:,.2f}</div>
            <p style="margin-top:10px;">Estimasi bersifat indikatif berdasarkan model machine learning.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("✨ Aplikasi prediksi harga rumah")
