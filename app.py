import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
    )
import matplotlib.pyplot as plt

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
    menu = st.selectbox("", ["Beranda", "Motif", "Klasifikasi", "Klasifikasi Banyak Gambar", "Riwayat"])

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
    "barong":"Motif barong melambangkan kekuatan, keberanian, perlindungan dari kejahatan, dan wibawa. Motif ini berakar dari kepercayaan akan keseimbangan antara kebaikan dan keburukan, serta sering dikaitkan dengan pemimpin yang bijaksana dan semangat pantang menyerah",
    "celup": "Motif celup atau jumputan, yang berkembang dari pengaruh teknik ikat Tiongkok dan India, secara simbolis melambangkan kesatuan, kreativitas, dan keragaman.",
    "cendrawasih": "Batik Cendrawasih adalah motif khas Papua yang terinspirasi dari burung endemik Bird of Paradise melambangkan keanggunan, keindahan alam Papua, spiritualitas, serta kebebasan.",
    "ceplok": "batik ceplok, yang merupakan salah satu motif tertua di Indonesia, melambangkan keteraturan hidup, keseimbangan, dan suratan takdir. Dengan pola geometris simetris seperti lingkaran atau mawar, motif ini mencerminkan harmoni empat arah mata angin dan ketentraman keluarga.",
    "dayak": "batik dayak melambangkan keharmonisan hubungan manusia dengan alam, Tuhan, dan sesama, serta keberanian. Motifnya terinspirasi kekayaan alam Borneo, seperti burung enggang (perdamaian/kemuliaan) dan sulur tanaman, serta sering menggambarkan aktivitas sungai.",
    "insang": "Batik atau Tenun Corak Insang adalah warisan budaya khas Pontianak, Kalimantan Barat, yang berasal dari masa Kesultanan Pontianak. Motif ini melambangkan napas kehidupan, kedekatan masyarakat Melayu dengan Sungai Kapuas, serta rasa syukur, keanggunan, dan dinamika kehidupan yang terus berubah.",
    "kawung": "Batik Kawung adalah salah satu motif tertua asal Yogyakarta (abad ke-13/16) yang terinspirasi dari buah aren atau kolang-kaling, melambangkan pengendalian diri yang sempurna, kesucian hati, keadilan, dan kesederhanaan",
    "lontara": "Batik Lontara, yang berasal dari Sulawesi Selatan, melambangkan identitas, kebanggaan, dan nilai-nilai luhur budaya Bugis-Makassar. Motifnya menggunakan aksara kuno Lontara, mencerminkan karakter jujur, kebenaran (lebih baik patah daripada bengkok), serta kekayaan alam",
    "mataketeran": "Batik Mata Keteran adalah salah satu motif batik khas dari Pamekasan, Madura. Motif ini mengambil inspirasi dari mata burung perkutut (Manok Keteran dalam bahasa Madura)",
    "megamendung": "Sejarah Batik Megamendung berasal dari Cirebon, Jawa Barat, yang lahir dari akulturasi budaya Tionghoa (awan) dan lokal. Motif ini melambangkan kesabaran, keteduhan, kesuburan, dan kehidupan. Warna biru dan gumpalan awan mendung menandakan langit luas yang tenang dan pembawa berkah",
    "ondel-ondel": "Batik Ondel-ondel merupakan motif khas Betawi yang melambangkan perlindungan, penolak bala (bahaya/wabah), serta harapan akan kehidupan yang lebih makmur dan aman.",
    "parang": "Batik Parang, salah satu motif tertua dari Kerajaan Mataram, melambangkan kekuatan, kewibawaan, perjuangan tanpa henti, dan kesinambungan. Motif diagonal seperti ombak laut ini menggambarkan perjalanan hidup yang penuh tantangan, mendorong pemakainya untuk terus berusaha memperbaiki diri, berani, dan bijaksana, serta dulunya merupakan simbol sakral kepemimpinan raja.",
    "pring": "Batik Pring Sedapur, khas Magetan, melambangkan persatuan, kebersamaan, kekuatan, dan kerendahan hati. Motif ini terinspirasi dari rumpun bambu yang tumbuh bersama, mencerminkan kehidupan yang rukun, saling menjaga, serta ketahanan karakter. Pring (bambu) juga bermakna adaptabilitas, sedangkan sedapur (satu dapur) berarti kerukunan keluarga.",
    "rumah-minang": "Batik Rumah Minang (termasuk batik tanah liek) berakar dari kebudayaan Minangkabau abad ke-19, menggunakan tanah liat dan bahan alami sebagai pewarna. Motifnya terinspirasi dari alam, ukiran Rumah Gadang, dan flora-fauna, yang melambangkan filosofi alam takambang jadi guru (alam bentang ilmu tempat manusia berguru) serta kekayaan budaya.",
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


    
    batik_image_path = os.path.join("assets", "batik.jpg")

    if os.path.exists(batik_image_path):
        st.image(batik_image_path, width="stretch")
    else:
        st.warning("Gambar batik tidak ditemukan")

    st.markdown("### Deskripsi Sistem")
    st.info("""
Aplikasi ini dibuat khusus untuk mengklasifikasikan motif batik berdasarkan gambar yang diunggah oleh pengguna. 
Adapun model yang digunakan untuk klasifikasi gambar ini adalah Convolutional Neural Network (CNN) dengan arsitektur EfficientNetB0. 
Pada dataset motif batik Indonesia, terdapat 14 motif batik yang dapat diklasifikasikan yaitu batik barong, batik celup, batik cendrawasih, batik ceplok, batik dayak, batik insang, batik kawung, batik lontara, batik mataketeran, batik megamendung, batik ondel-ondel, batik parang, batik pring, dan batik rumah-minang.
""")

    st.markdown("### Cara Menggunakan")
    st.success("""
Upload gambar batik → sistem klasifikasi → hasil muncul → otomatis tersimpan di riwayat.
""")

# ===========================
# MOTIF
# ===========================
elif menu == "Motif":
    st.markdown("<div class='title'>Galeri Motif Batik</div>", unsafe_allow_html=True)

    cols = st.columns(4)
    for i, name in enumerate(class_names):
        with cols[i % 4]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            path = os.path.join("assets", name + ".jpg")
            if os.path.exists(path):
                st.image(path, width="stretch")
            else:
                st.warning("Tidak ada gambar")

            st.markdown(name.title(), unsafe_allow_html=True)
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
            st.image(img, width="stretch")

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
                 # ===========================
                # ✅ TAMBAHAN DESKRIPSI
                # ===========================
                deskripsi = deskripsi_motif.get(label.lower(), "Deskripsi belum tersedia.")
                st.markdown(f"""
                <div style='background:#f0fdf4; padding:15px; border-radius:10px; margin-top:10px'>
                <b>Deskripsi:</b><br>{deskripsi}
                </div>
                """, unsafe_allow_html=True)

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
                "Klasifikasi": label,
                "Confidence": f"{conf*100:.2f}%",
                "Gambar": img.copy()
            })
# ===========================
# KLASIFIKASI BANYAK GAMBAR
# ===========================
elif menu == "Klasifikasi Banyak Gambar":

    import zipfile
    import tempfile

    st.markdown(
        "<div class='title'>Klasifikasi Dataset Batik (.zip)</div>",
        unsafe_allow_html=True
    )

    uploaded_zip = st.file_uploader(
        "Upload Dataset ZIP",
        type=["zip"]
    )

    if uploaded_zip is not None:

        hasil = []

        with tempfile.TemporaryDirectory() as temp_dir:

            # Simpan file zip
            zip_path = os.path.join(temp_dir, "dataset.zip")

            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())

            # Ekstrak zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Cari semua gambar
            image_files = []
            y_true = []
            y_pred = []

            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_files.append(os.path.join(root, file))

            if len(image_files) == 0:
                st.error("Tidak ada gambar ditemukan dalam file ZIP.")

            else:

                st.success(f"{len(image_files)} gambar ditemukan.")

                progress = st.progress(0)

                for i, path in enumerate(image_files):
                    try:
                        true_label = os.path.basename(os.path.dirname(path))
                        img = Image.open(path).convert("RGB")
                        pred = predict(img)
                        idx = np.argmax(pred)
                        conf = float(pred[idx])
                        threshold = 0.6
                        if conf >= threshold:
                            label = class_names[idx]
                        else:
                            results = find_similar(img)
                            label = results[0][0] if results else "Tidak dikenali"
                            y_true.append(true_label)
                            y_pred.append(label)
                            hasil.append({
                                "Nama File": os.path.basename(path),
                                "Label Asli": true_label,
                                "Prediksi": label,
                                "Confidence (%)": round(conf*100,2)
                                })
                            st.session_state.history.append({
                                "Waktu": datetime.now().strftime("%H:%M:%S"),
                                "File": os.path.basename(path),
                                "Klasifikasi": label,
                                "Confidence": f"{conf*100:.2f}%",
                                "Gambar": img.copy()
                                })

                    except Exception as e:

                        hasil.append({
                            "Nama File": os.path.basename(path),
                            "Motif": "Error",
                            "Confidence (%)": "-"
                        })

                    progress.progress((i+1)/len(image_files))

                st.success("Klasifikasi selesai.")

                df_hasil = pd.DataFrame(hasil)

                st.dataframe(
                    df_hasil,
                    width="stretch"
                )

                csv = df_hasil.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇ Download Hasil CSV",
                    csv,
                    file_name="hasil_klasifikasi_dataset.csv",
                    mime="text/csv"
                )
                st.divider()
                st.subheader("Evaluasi Model")
                accuracy = accuracy_score(y_true, y_pred)
                st.metric(
                    "Accuracy",
                    f"{accuracy*100:.2f}%"
                    )
                report = classification_report(
                    y_true,
                    y_pred,
                    labels=class_names,
                    output_dict=True,
                    zero_division=0
                    )
                report_df = pd.DataFrame(report).transpose()
                st.subheader("Classification Report")
                st.dataframe(
                    report_df,
                    width="stretch"
                    )
                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=class_names
                    )
                cm_df = pd.DataFrame(
                    cm,
                    index=class_names,
                    columns=class_names
                    )
                st.subheader("Confusion Matrix")
                st.dataframe(
                    cm_df,
                    width="stretch"
                    )
                fig, ax = plt.subplots(figsize=(12,10))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks(range(len(class_names)))
                ax.set_yticks(range(len(class_names)))
                ax.set_xticklabels(class_names, rotation=90)
                ax.set_yticklabels(class_names)
                plt.xlabel("Prediksi")
                plt.ylabel("Label Asli")
                plt.title("Confusion Matrix")
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        ax.text(
                            j,
                            i,
                            cm[i, j],
                            ha="center",
                            va="center",
                            fontsize=7
                            )
                        plt.colorbar(im)
                        st.pyplot(fig)
                        st.subheader("Statistik Prediksi")
                        statistik = (
                            df_hasil["Prediksi"]
                            .value_counts()
                            .reset_index()
                            )
                        statistik.columns = ["Motif", "Jumlah"]
                        st.dataframe(statistik, width="stretch")
                        st.bar_chart(
                            statistik.set_index("Motif")
                            )

# ===========================
# RIWAYAT (MODERN)
# ===========================
elif menu == "Riwayat":
    st.markdown("<div class='title'>Riwayat</div>", unsafe_allow_html=True)

    if st.session_state.history:
        for item in st.session_state.history[::-1]:

            col1, col2 = st.columns([1,4])

            with col1:
                st.image(item["Gambar"], width="stretch")

            with col2:
                st.markdown(f"""
                <div class="history-card">
                    <div class="history-label">{item['Klasifikasi'].upper()}</div>
                    <div class="history-meta">File: {item['File']}</div>
                    <div class="history-meta">Waktu: {item['Waktu']}</div>
                    <div class="history-badge">Confidence: {item['Confidence']}</div>
                </div>
                """, unsafe_allow_html=True)

        df = pd.DataFrame([
            {k:v for k,v in item.items() if k != "Gambar"}
            for item in st.session_state.history
        ])

        st.download_button("⬇ Download CSV", df.to_csv(index=False), "riwayat.csv")

        if st.button("🗑 Hapus Riwayat"):
            st.session_state.history = []
            st.success("Riwayat dihapus")

    else:
        st.info("Belum ada data")

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
