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
# HEADER (TAMBAHAN)
# ===========================
st.markdown("""
<div style='text-align:center; margin-bottom:20px;'>
<h2>🧵 Batik AI Recognition System</h2>
<p style='opacity:0.7'>Klasifikasi motif batik berbasis Deep Learning</p>
</div>
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
}
.badge {
    background: #16a34a;
    padding:6px 16px;
    border-radius:999px;
    color:white;
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
# DESKRIPSI (TAMBAHAN)
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

    st.markdown("### Cara Menggunakan")
    c1,c2,c3 = st.columns(3)
    c1.info("1. Upload gambar")
    c2.info("2. Sistem analisis")
    c3.info("3. Lihat hasil")

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
# KLASIFIKASI
# ===========================
elif menu == "Klasifikasi":
    st.markdown("<div class='title'>Klasifikasi</div>", unsafe_allow_html=True)

    file = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).convert("RGB")

        col1, col2 = st.columns([1,2])

        with col1:
            st.image(img)

        with col2:
            st.markdown("### Hasil Analisis")

            pred = predict(img)
            idx = np.argmax(pred)
            conf = float(pred[idx])

            if conf >= 0.6:
                label = class_names[idx]

                st.markdown(f"<div class='badge'>{label.upper()}</div>", unsafe_allow_html=True)
                st.write(f"{conf*100:.2f}%")
                st.progress(conf)

                if label in deskripsi_motif:
                    st.success(deskripsi_motif[label])

            else:
                st.warning("Menggunakan similarity")
                results = find_similar(img)

                for l,p,s in results:
                    st.image(p, width=120)
                    st.write(f"{l} ({s*100:.2f}%)")

                label = results[0][0] if results else "Tidak dikenali"

            st.session_state.history.append({
                "Waktu": datetime.now().strftime("%H:%M:%S"),
                "File": file.name,
                "Klasifikasi": label,
                "Confidence": f"{conf*100:.2f}%",
                "Gambar": img.copy()
            })

# ===========================
# RIWAYAT
# ===========================
elif menu == "Riwayat":
    st.markdown("<div class='title'>Riwayat</div>", unsafe_allow_html=True)

    if st.session_state.history:
        for item in st.session_state.history[::-1]:
            col1, col2 = st.columns([1,3])

            with col1:
                st.image(item["Gambar"])

            with col2:
                st.write(item["Waktu"], item["Klasifikasi"], item["Confidence"])

        df = pd.DataFrame([
            {k:v for k,v in item.items() if k != "Gambar"}
            for item in st.session_state.history
        ])

        st.download_button("Download CSV", df.to_csv(index=False), "riwayat.csv")

# ===========================
# FOOTER
# ===========================
st.markdown("""
<hr>
<p style='text-align:center; font-size:12px; opacity:0.6'>
© 2026 Sistem Klasifikasi Batik
</p>
""", unsafe_allow_html=True)
