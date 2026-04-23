import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
from sklearn.metrics.pairwise import cosine_similarity

# ===========================
# CONFIG
# ===========================
st.set_page_config(layout="wide", page_title="Batik AI")

# ===========================
# HEADER
# ===========================
st.markdown("""

""", unsafe_allow_html=True)

# ===========================
# SESSION
# ===========================
if "history" not in st.session_state:
    st.session_state.history = []

# ===========================
# STYLE
# ===========================
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(150deg, #81d4fa, #0284c7) !important;
}
.title {
    font-size: 30px;
    font-weight:700;
    text-align:center;
}
.card {
    background: rgba(255,255,255,0.9);
    border-radius:15px;
    padding:10px;
    text-align:center;
    transition: 0.3s;
}

/* 🔥 HOVER EFFECT */
.card:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}

.badge {
    background: #16a34a;
    padding:6px 16px;
    border-radius:999px;
    color:white;
}

/* RIWAYAT */
.history-card {
    background: white;
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.history-label {
    font-weight: 700;
    font-size: 16px;
}
.history-meta {
    font-size: 13px;
    opacity: 0.7;
}
.history-badge {
    background: #16a34a;
    color: white;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    display: inline-block;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# MENU
# ===========================
with st.sidebar:
    menu = st.selectbox("", ["Beranda", "Motif", "Klasifikasi", "Riwayat"])

# ===========================
# MODEL
# ===========================
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

@st.cache_resource
def load_model():
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(14, activation='softmax')
    ])

    model.load_weights("model_efficientnet.keras")
    return model

model = load_model()

# ===========================
# FEATURE EXTRACTOR
# ===========================
@st.cache_resource
def get_feature_extractor():
    dummy = np.zeros((1,224,224,3))
    model.predict(dummy)
    return tf.keras.Model(inputs=model.inputs, outputs=model.layers[-3].output)

feature_extractor = get_feature_extractor()

# ===========================
# DATABASE
# ===========================
@st.cache_resource
def load_database():
    import gdown, zipfile

    if not os.path.exists("dataset_similarity"):
        gdown.download(id="1JoxAUD7ciykkPTRr3wkIL_aG3mZPI8vq", output="dataset.zip", quiet=False)
        with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
            zip_ref.extractall()

    features, labels, paths = [], [], []

    for label in os.listdir("dataset_similarity"):
        folder = os.path.join("dataset_similarity", label)
        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            try:
                img = Image.open(path).convert("RGB").resize((224,224))
                arr = preprocess_input(np.array(img))
                arr = np.expand_dims(arr, axis=0)
                feat = feature_extractor.predict(arr)[0]
                features.append(feat)
                labels.append(label)
                paths.append(path)
            except:
                pass

    return np.array(features), labels, paths

db_features, db_labels, db_paths = load_database()

# ===========================
# SIMILARITY
# ===========================
def find_similar(img):
    img = img.resize((224,224))
    arr = preprocess_input(np.array(img))
    arr = np.expand_dims(arr, axis=0)

    query_feat = feature_extractor.predict(arr)
    sim = cosine_similarity(query_feat, db_features)[0]
    idx = np.argsort(sim)[-3:][::-1]

    return [(db_labels[i], db_paths[i], sim[i]) for i in idx]

# ===========================
# CLASS
# ===========================
class_names = [
    'barong','celup','cendrawasih','ceplok','dayak','insang',
    'kawung','lontara','mataketeran','megamendung','ondel-ondel',
    'parang','pring','rumah-minang'
]

# ===========================
# DESKRIPSI
# ===========================
deskripsi_motif = {
    "parang": "Melambangkan kekuatan dan kesinambungan.",
    "kawung": "Melambangkan kesucian dan keadilan.",
    "megamendung": "Melambangkan ketenangan dan kesabaran.",
}

# ===========================
# PREDICT
# ===========================
def predict(img):
    img = img.resize((224,224))
    arr = preprocess_input(np.array(img))
    arr = np.expand_dims(arr, axis=0)
    return model.predict(arr)[0]

# ===========================
# BERANDA
# ===========================
if menu == "Beranda":
    st.markdown("<div class='title'>Sistem Klasifikasi Motif Batik</div>", unsafe_allow_html=True)

    # ✅ BANNER
    st.image("https://www.pinterest.com/pin/36-gambar-mentahan-transparan-bunga-batik-png--961659326658235243/", use_column_width=True)

    st.markdown("### Deskripsi Sistem")
    st.info("""
Aplikasi ini dibuat khusus untuk mengklasifikasikan motif batik berdasarkan gambar yang diunggah oleh pengguna. 
Adapun model yang digunakan adalah CNN EfficientNetB0.
""")

    st.markdown("### Cara Menggunakan")
    st.success("""
Upload gambar batik → sistem klasifikasi → hasil muncul → otomatis tersimpan di riwayat.
""")

# ===========================
# MOTIF
# ===========================
elif menu == "Motif":
    st.markdown("<div class='title'>Galeri Motif</div>", unsafe_allow_html=True)

    cols = st.columns(4)
    for i, name in enumerate(class_names):
        with cols[i % 4]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            path = os.path.join("assets", name + ".jpg")
            if os.path.exists(path):
                st.image(path, use_column_width=True)
            else:
                st.warning("Tidak ada gambar")

            st.markdown(name.title(), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# KLASIFIKASI & RIWAYAT (TIDAK DIUBAH)
# ===========================

# ===========================
# FOOTER (UPGRADE)
# ===========================
st.markdown("""
<hr>
<div style='text-align:center; padding:10px'>
    <b>🎓 Sistem Klasifikasi Motif Batik</b><br>
    Menggunakan Deep Learning CNN EfficientNetB0<br><br>
    <span style='font-size:12px; opacity:0.6'>
    © 2026 | Skripsi AI Computer Vision
    </span>
</div>
""", unsafe_allow_html=True)
