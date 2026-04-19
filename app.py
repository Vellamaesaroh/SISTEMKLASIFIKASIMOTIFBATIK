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
    font-size: 32px;
    font-weight:700;
    text-align:center;
}
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
# MENU
# ===========================
with st.sidebar:
    menu = st.selectbox("Menu", ["Beranda", "Klasifikasi", "Riwayat"])

# ===========================
# LOAD MODEL
# ===========================
import keras

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

@st.cache_resource
def load_model():
    try:
        # ==========================
        # REBUILD MODEL
        # ==========================
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224,224,3)
        )
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(14, activation='softmax')
        ])

        # ==========================
        # LOAD WEIGHTS (COBA)
        # ==========================
        model.load_weights("model_efficientnet.keras")

        return model

    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

model = load_model()

# ===========================
# FEATURE EXTRACTOR
# ===========================
@st.cache_resource
def get_feature_extractor():
    return tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-3].output
    )

feature_extractor = get_feature_extractor()

# ===========================
# LOAD DATASET (AUTO DOWNLOAD)
# ===========================
@st.cache_resource
def load_database():
    import gdown
    import zipfile

    dataset_path = "dataset_similarity"

    if not os.path.exists(dataset_path):

        FILE_ID = "MASUKKAN_FILE_ID_DRIVE_KAMU"
        url = f"https://drive.google.com/uc?id={FILE_ID}"

        output = "dataset.zip"

        st.info("Downloading dataset...")
        gdown.download(url, output, quiet=False)

        st.info("Extracting dataset...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall()

    features = []
    labels = []
    paths = []

    for label in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, label)

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
# SIMILARITY FUNCTION
# ===========================
def find_similar(img, top_k=3):
    if len(db_features) == 0:
        return []

    img = img.resize((224,224))
    arr = preprocess_input(np.array(img))
    arr = np.expand_dims(arr, axis=0)

    query_feat = feature_extractor.predict(arr)

    sim = cosine_similarity(query_feat, db_features)[0]
    idx = np.argsort(sim)[-top_k:][::-1]

    return [(db_labels[i], db_paths[i], sim[i]) for i in idx]

# ===========================
# CLASS MODEL
# ===========================
class_names = [
    'barong','celup','cendrawasih','ceplok','dayak','insang',
    'kawung','lontara','mataketeran','megamendung','ondel-ondel',
    'parang','pring','rumah-minang'
]

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

# ===========================
# KLASIFIKASI
# ===========================
elif menu == "Klasifikasi":

    st.markdown("<div class='title'>Klasifikasi Motif Batik</div>", unsafe_allow_html=True)

    file = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).convert("RGB")

        col1, col2 = st.columns([1,2])

        with col1:
            st.image(img, use_column_width=True)

        with col2:
            pred = predict(img)
            idx = np.argmax(pred)
            conf = float(pred[idx])

            threshold = 0.6

            if conf >= threshold:
                label = class_names[idx]
                st.markdown(f"<div class='badge'>{label.upper()}</div>", unsafe_allow_html=True)
                st.write(f"Confidence: {conf*100:.2f}%")
                st.progress(conf)

            else:
                st.warning("Motif tidak dikenali → pakai similarity")

                results = find_similar(img)

                for l, p, s in results:
                    st.image(p, width=150)
                    st.write(f"{l} ({s*100:.2f}%)")

                label = results[0][0] if results else "Tidak dikenali"

            # riwayat
            st.session_state.history.append({
                "Waktu": datetime.now().strftime("%H:%M:%S"),
                "File": file.name,
                "Prediksi": label,
                "Confidence": f"{conf*100:.2f}%",
                "Gambar": img.copy()
            })

# ===========================
# RIWAYAT
# ===========================
elif menu == "Riwayat":

    st.markdown("<div class='title'>Riwayat</div>", unsafe_allow_html=True)

    for item in st.session_state.history[::-1]:
        col1, col2 = st.columns([1,3])

        with col1:
            st.image(item["Gambar"], use_column_width=True)

        with col2:
            st.write(item["Waktu"])
            st.write(item["File"])
            st.write(item["Prediksi"])
            st.write(item["Confidence"])
            st.markdown("---")

    if st.button("Hapus Riwayat"):
        st.session_state.history = []
