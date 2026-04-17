import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import os
import base64

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
# FUNCTION BASE64 IMAGE
# ===========================
def get_base64_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_img = get_base64_image("assets/bg_2.jpg")

# ===========================
# STYLE
# ===========================
st.markdown(f"""
<style>

/* SIDEBAR BACKGROUND */
section[data-testid="stSidebar"] {{
    background: url("data:image/jpg;base64,{bg_img}");
    background-size: 250px;
    background-position: center;
}}

section[data-testid="stSidebar"]::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: rgba(0,0,0,0.4);
}}

section[data-testid="stSidebar"] .block-container {{
    position: relative;
    z-index: 1;
}}

.title {{
    font-size: clamp(10px, 4vw, 32px);
    font-weight:700;
    text-align:center;
}}

.subtitle {{
    text-align:center;
    opacity:0.7;
    margin-bottom:20px;
}}

.card {{
    background: rgba(255,255,255,0.85);
    border-radius:16px;
    padding:15px;
    text-align:center;
}}

.badge {{
    background: #16a34a;
    padding:6px 16px;
    border-radius:999px;
    color:white;
    font-weight:600;
}}

</style>
""", unsafe_allow_html=True)

# ===========================
# MENU
# ===========================
with st.sidebar:
    st.markdown("<h4>MENU</h4>", unsafe_allow_html=True)
    menu = st.selectbox("", ["Beranda", "Motif", "Klasifikasi", "Riwayat"])

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
# DUMMY PREDICT (OPS 2)
# ===========================
def predict(img):
    pred = np.random.rand(len(class_names))
    pred = pred / np.sum(pred)
    return pred

# ===========================
# BERANDA
# ===========================
if menu == "Beranda":
    st.markdown("<div class='title'>Sistem Klasifikasi Motif Batik</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='subtitle'>"
        "Aplikasi AI untuk klasifikasi motif batik secara otomatis"
        "</div>",
        unsafe_allow_html=True
    )

    path = os.path.join("assets", "batik.jpg")
    if os.path.exists(path):
        st.image(path, use_column_width=True)

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

            if img_path:
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

    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg","png","jpeg"])

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
    st.markdown("<div class='title'>Riwayat</div>", unsafe_allow_html=True)

    if st.session_state.history:
        for item in st.session_state.history[::-1]:
            col1, col2 = st.columns([1,3])

            with col1:
                st.image(item["Gambar"], use_column_width=True)

            with col2:
                st.write(item)

        df = pd.DataFrame([
            {k:v for k,v in item.items() if k != "Gambar"}
            for item in st.session_state.history
        ])

        st.download_button("Download CSV", df.to_csv(index=False), "riwayat.csv")

        if st.button("Hapus Riwayat"):
            st.session_state.history = []
            st.success("Riwayat dihapus")
    else:
        st.info("Belum ada data")
