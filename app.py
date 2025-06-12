import streamlit as st
st.set_page_config(page_title="Prediksi Harga Rumah Jakarta", page_icon="🏠", layout="centered")

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ----------------------------
# Setup awal model
# ----------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv('dataset/Data_Lamudi_Cleaned.csv')

    fitur = ['Kamar Tidur', 'Kamar Mandi', 'Luas Bangunan', 'Luas Tanah']
    X = df[fitur]
    y = df['Harga']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler

model, scaler = train_model()

# ----------------------------
# UI Streamlit
# ----------------------------
st.markdown("<h1 style='text-align: center;'>🏠 Prediksi Harga Rumah di Jakarta</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Gunakan input di bawah ini untuk memprediksi harga rumah</p>", unsafe_allow_html=True)

st.subheader("📝 Masukkan Informasi Properti")
kamar_tidur = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, value=3, step=1)
kamar_mandi = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=10, value=2, step=1)
luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=20, max_value=500, value=120, step=1)
luas_tanah = st.number_input("Luas Tanah (m²)", min_value=20, max_value=600, value=100, step=1)

if st.button("🔍 Prediksi Harga"):
    input_data = pd.DataFrame([{
        "Kamar Tidur": kamar_tidur,
        "Kamar Mandi": kamar_mandi,
        "Luas Bangunan": luas_bangunan,
        "Luas Tanah": luas_tanah
    }])
    input_scaled = scaler.transform(input_data)
    prediksi_harga = model.predict(input_scaled)[0]

    st.subheader("💰 Estimasi Harga Rumah:")
    st.metric(label="Harga Perkiraan", value=f"Rp {prediksi_harga:,.0f}")

st.markdown("---")
st.markdown("<small>© 2025 | Aplikasi oleh Assetify</small>", unsafe_allow_html=True)
