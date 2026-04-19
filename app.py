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
# FEATURE EXTRACTOR (FIX ERROR)
# ===========================
@st.cache_resource
def get_feature_extractor():
    dummy = np.zeros((1,224,224,3))
    model.predict(dummy)

    return tf.keras.Model(
        inputs=model.inputs,
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

        FILE_ID = "1JoxAUD7ciykkPTRr3wkIL_aG3mZPI8vq"
        output = "dataset.zip"

        
        gdown.download(id=FILE_ID, output=output, quiet=False)

        
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
# SIMILARITY
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
    arr = preprocess_input(np.array(img))
    arr = np.expand_dims(arr, axis=0)
    return model.predict(arr)[0]

# ===========================
# BERANDA
# ===========================
if menu == "Beranda":
    st.markdown("<div class='title'>Sistem Klasifikasi Motif Batik</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>"
        "Selamat datang di Sistem Klasifikasi Motif Batik, sebuah aplikasi berbasis kecerdasan buatan <br>"
        "yang dirancang untuk mengenali dan mengklasifikasikan motif batik secara otomatis."
        "</div>",
        unsafe_allow_html=True
    )

    if os.path.exists("assets/batik.jpg"):
        st.image("assets/batik.jpg", use_column_width=True)

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
