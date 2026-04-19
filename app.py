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
# SESSION STATE
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
.subtitle {
    text-align:center;
    opacity:0.7;
}
.card {
    background: rgba(255,255,255,0.85);
    border-radius:16px;
    padding:15px;
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
    st.markdown("<h4>MENU</h4>", unsafe_allow_html=True)
    menu = st.selectbox("", ["Beranda", "Motif", "Klasifikasi", "Riwayat"])

# ===========================
# LOAD MODEL (AMAN)
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
# LOAD DATASET SIMILARITY
# ===========================
@st.cache_resource
def load_database():
    features = []
    labels = []
    paths = []

    import gdown

url = "https://drive.google.com/drive/folders/1EjZtYjFsClR4NzTYSs1vS-Or-rat9sff?usp=drive_link"
output = "dataset.zip"

gdown.download(url, output, quiet=False)

    if not os.path.exists(dataset_path):
        return np.array([]), [], []

    for label in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, label)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((224,224))

                arr = np.array(img)
                arr = preprocess_input(arr)
                arr = np.expand_dims(arr, axis=0)

                feat = feature_extractor.predict(arr)[0]

                features.append(feat)
                labels.append(label.lower())
                paths.append(img_path)

            except:
                pass

    return np.array(features), labels, paths

db_features, db_labels, db_paths = load_database()

# ===========================
# FUNCTION SIMILARITY
# ===========================
def find_similar(img, top_k=3):
    if len(db_features) == 0:
        return []

    img = img.resize((224,224))
    arr = np.array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    query_feat = feature_extractor.predict(arr)

    sim = cosine_similarity(query_feat, db_features)[0]
    idxs = np.argsort(sim)[-top_k:][::-1]

    results = []
    for i in idxs:
        results.append((db_labels[i], db_paths[i], sim[i]))

    return results

# ===========================
# CLASS (MODEL UTAMA)
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
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]
    return pred

# ===========================
# BERANDA
# ===========================
if menu == "Beranda":
    st.markdown("<div class='title'>Sistem Klasifikasi Motif Batik</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Aplikasi AI untuk klasifikasi motif batik</div>", unsafe_allow_html=True)

# ===========================
# KLASIFIKASI
# ===========================
elif menu == "Klasifikasi":
    st.markdown("<div class='title'>Klasifikasi Motif Batik</div>", unsafe_allow_html=True)

    if model is None:
        st.error("Model tidak tersedia")
    else:
        st.success("Model siap digunakan")

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
                conf = float(pred[idx])
                threshold = 0.6

                # ======================
                # JIKA YAKIN
                # ======================
                if conf >= threshold:
                    label = class_names[idx]
                    st.markdown(f"<div class='badge'>{label.upper()}</div>", unsafe_allow_html=True)

                    st.write(f"Confidence: {conf*100:.2f}%")
                    st.progress(conf)

                # ======================
                # JIKA RAGU → SIMILARITY
                # ======================
                else:
                    results = find_similar(img)

                    st.warning("Motif tidak ada di dataset utama")
                    st.subheader("Motif Paling Mirip")

                    for label_sim, path_sim, score in results:
                        st.image(path_sim, width=150)
                        st.write(f"{label_sim.title()} ({score*100:.2f}%)")

                    label = f"Mirip: {results[0][0]}" if results else "Tidak Dikenali"

                # ======================
                # TOP 3 MODEL
                # ======================
                st.subheader("Top 3 Prediksi Model")
                top3 = pred.argsort()[-3:][::-1]

                for i in top3:
                    st.write(f"{class_names[i]}: {pred[i]*100:.2f}%")

                # ======================
                # SIMPAN RIWAYAT
                # ======================
                st.session_state.history.append({
                    "Waktu": datetime.now().strftime("%H:%M:%S"),
                    "File": uploaded_file.name,
                    "Prediksi": label,
                    "Confidence": f"{conf*100:.2f}%",
                    "Gambar": img.copy()
                })

# ===========================
# RIWAYAT
# ===========================
elif menu == "Riwayat":
    st.markdown("<div class='title'>Riwayat Klasifikasi</div>", unsafe_allow_html=True)

    if st.session_state.history:
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
