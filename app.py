import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# ===========================
# CONFIG
# ===========================
st.set_page_config(layout="wide", page_title="Batik")

# ===========================
# SESSION STATE
# ===========================
if "history" not in st.session_state:
    st.session_state.history = []

# ===========================
# STYLE (FINAL FIX RESPONSIVE)
# ===========================
st.markdown("""
<style>

/* SIDEBAR BACKGROUND */
section[data-testid="stSidebar"] {
    background: linear-gradient(150deg, #81d4fa, #0284c7) !important;
}

/* JUDUL MENU */
section[data-testid="stSidebar"] h4 {
    color: white;
    font-weight: 700;
    margin-bottom: 15px;
}

/* SELECTBOX */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background: white;
    border-radius: 10px;
    padding: 5px;
}

/* TEXT DALAM DROPDOWN */
section[data-testid="stSidebar"] .stSelectbox div {
    color: #0f172a;
    font-weight: 500;
}

/* HOVER DROPDOWN */
section[data-testid="stSidebar"] .stSelectbox div:hover {
    background: #e0f2fe;
}

/* TITLE */
.title {
    font-size: clamp(10px, 4vw, 32px);
    font-weight:700;
    text-align:center;
    margin-bottom:10px;
    word-wrap: break-word;
    line-height:1.3;
}

/* SUBTITLE */
.subtitle {
    text-align:center;
    opacity:0.7;
    margin-bottom:20px;
    padding: 0 10px;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.85);
    border-radius:16px;
    padding:15px;
    text-align:center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

/* BADGE */
.badge {
    background: #16a34a;
    padding:6px 16px;
    border-radius:999px;
    color:white;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# ===========================
# MENU (DROPDOWN LIKE GAMBAR)
# ===========================
with st.sidebar:
    st.markdown("<h4>MENU</h4>", unsafe_allow_html=True)

    menu = st.selectbox(
        "",
        ["Beranda", "Motif", "Klasifikasi", "Riwayat"]
    )

# ===========================
# LOAD MODEL
# ===========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_batik_effnet")
    return model.signatures["serving_default"]

model = load_model()

# ===========================
# CLASS
# ===========================
class_names = [
    'barong','celup','cendrawasih','ceplok','dayak','insang',
    'kawung','lontara','mataketeran','megamendung','ondel-ondel',
    'parang','pring','rumah-minang'
]

# ===========================
# LOAD GAMBAR
# ===========================
image_folder = os.path.abspath("assets")

category_images = {}
for name in class_names:
    found = None
    for ext in [".jpg", ".png", ".jpeg"]:
        path = os.path.join(image_folder, name + ext)
        if os.path.isfile(path):
            found = path
            break
    category_images[name] = found

# ===========================
# PREDICT
# ===========================
def predict(img):
    img = img.resize((224,224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred_tensor = model(tf.convert_to_tensor(img_array))
    pred = list(pred_tensor.values())[0].numpy()[0]

    return pred

# ===========================
# BERANDA
# ===========================
if menu == "Beranda":
    st.markdown("<div class='title'>Sistem Klasifikasi Motif Batik</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='subtitle'>"
        "Selamat datang di Sistem Klasifikasi Motif Batik, sebuah aplikasi berbasis kecerdasan buatan <br>"
        "yang dirancang untuk mengenali dan mengklasifikasikan motif batik Indonesia secara otomatis."
        "</div>",
        unsafe_allow_html=True
    )

    batik_image_path = os.path.join("assets", "batik.jpg")

    if os.path.exists(batik_image_path):
        st.image(batik_image_path, use_column_width=True)
    else:
        st.warning("Gambar batik tidak ditemukan")

# ===========================
# MOTIF
# ===========================
elif menu == "Motif":
    st.markdown("<div class='title'>Galeri Motif Batik</div>", unsafe_allow_html=True)

    cols = st.columns(4)

    for i, name in enumerate(class_names):
        with cols[i % 4]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            img_path = category_images.get(name)

            if img_path and os.path.exists(img_path):
                st.image(Image.open(img_path), use_column_width=True)
            else:
                st.warning("Gambar tidak tersedia")

            st.markdown(f"<b>{name.title()}</b>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# KLASIFIKASI
# ===========================
elif menu == "Klasifikasi":
    st.markdown("<div class='title'>Klasifikasi Motif Batik</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Gambar Batik", type=["jpg","png","jpeg"])

    if uploaded_file:
        col1, col2 = st.columns([1,2])

        with col1:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, use_column_width=True)

        with col2:
            with st.spinner("Memproses..."):
                pred = predict(img)

            idx = np.argmax(pred)
            label = class_names[idx]
            conf = pred[idx]

            st.markdown(f"<div class='badge'>{label.upper()}</div>", unsafe_allow_html=True)
            st.write(f"Confidence: {conf*100:.2f}%")

            st.progress(float(conf))

            # ✅ TAMBAHAN SIMPAN GAMBAR
            st.session_state.history.append({
                "Waktu": datetime.now().strftime("%H:%M:%S"),
                "File": uploaded_file.name,
                "Prediksi": label,
                "Confidence": f"{conf*100:.2f}%",
                "Gambar": img.copy()
            })

            st.bar_chart(pred)

# ===========================
# RIWAYAT
# ===========================
elif menu == "Riwayat":
    st.markdown("<div class='title'>Riwayat Klasifikasi</div>", unsafe_allow_html=True)

    if st.session_state.history:

        # ✅ TAMPILKAN DENGAN GAMBAR
        for item in st.session_state.history[::-1]:
            col1, col2 = st.columns([1,3])

            with col1:
                st.image(item["Gambar"], use_column_width=True)

            with col2:
                st.markdown(f"**Waktu:** {item['Waktu']}")
                st.markdown(f"**File:** {item['File']}")
                st.markdown(f"**Prediksi:** {item['Prediksi']}")
                st.markdown(f"**Confidence:** {item['Confidence']}")
                st.markdown("---")

        # ✅ DOWNLOAD TANPA GAMBAR
        df = pd.DataFrame([
            {k:v for k,v in item.items() if k != "Gambar"}
            for item in st.session_state.history
        ])

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "riwayat.csv"
        )

        if st.button("Hapus Riwayat"):
            st.session_state.history = []
            st.success("Riwayat berhasil dihapus")
    else:
        st.info("Belum ada data riwayat")