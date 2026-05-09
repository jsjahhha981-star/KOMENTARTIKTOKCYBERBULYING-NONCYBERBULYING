import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Cyberbullying Detection",
    layout="wide"
)

# ======================
# LOAD MODEL
# ======================
model = joblib.load('MODEL_AMAN.pkl')

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}

/* MAIN TITLE */
.main-title {
    font-size: 52px;
    font-weight: 800;
    color: white;
    text-align: center;
    margin-bottom: 20px;
    line-height: 1.2;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: #f8fafc;
    font-size: 20px;
    line-height: 1.8;
    font-weight: 400;
}

/* HERO SECTION */
.hero {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    padding: 70px 50px;
    border-radius: 30px;
    margin-bottom: 35px;
    box-shadow: 0px 15px 35px rgba(0,0,0,0.18);
}

/* CARD */
.card {
    background: white;
    padding: 30px;
    border-radius: 22px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 25px;
    color: #1e293b;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1e3a8a 0%,
        #4338ca 50%,
        #7c3aed 100%
    );
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* RADIO BUTTON TEXT */
.stRadio label {
    color: white !important;
    font-size: 17px !important;
    font-weight: 500 !important;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(
        to right,
        #2563eb,
        #7c3aed
    );
    color: white;
    font-size: 17px;
    font-weight: bold;
    transition: 0.3s;
}

/* BUTTON HOVER */
.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(
        to right,
        #1d4ed8,
        #6d28d9
    );
    color: white;
}

/* INFO BOX */
.info-box {
    background: #eff6ff;
    padding: 22px;
    border-radius: 18px;
    border-left: 6px solid #2563eb;
    color: #1e293b;
    font-size: 16px;
    line-height: 1.7;
}

/* METRIC CARD */
.metric-card {
    background: white;
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.08);
    transition: 0.3s;
}

/* METRIC HOVER */
.metric-card:hover {
    transform: translateY(-5px);
}

/* METRIC NUMBER */
.metric-number {
    font-size: 42px;
    font-weight: 800;
    color: #2563eb;
    margin-bottom: 10px;
}

/* METRIC TEXT */
.metric-text {
    color: #475569;
    font-size: 17px;
}

/* TEXT AREA */
textarea {
    border-radius: 15px !important;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
st.sidebar.title(" MENU")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Beranda",
        "Prediksi",
        "Upload CSV / XLSX",
        "Evaluasi Model"
    ]
)

# ======================
# BERANDA
# ======================
if menu == "Beranda":

    st.markdown("""
    <div class="hero">
        <div class="main-title">
            Sistem Deteksi Cyberbullying Komentar TikTok
        </div>

        
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">AI</div>
            <div class="metric-text">
                Machine Learning Detection
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">SVM</div>
            <div class="metric-text">
                Support Vector Machine
            </div>
        </div>
        """, unsafe_allow_html=True)

    

    st.markdown("<br>", unsafe_allow_html=True)

    

    st.markdown("""
    <div class="card">
        <h2 style="color:#2563eb;">
            📌 Tentang Sistem
        </h2>
        <p style="font-size:17px; line-height:1.8; color:#334155;">
        Sistem ini dibuat untuk membantu mendeteksi komentar cyberbullying
        pada platform media sosial TikTok menggunakan teknologi
        Machine Learning. <br>Model menggunakan metode: <b>Support Vector Machine (SVM)</b><br><br>Sistem dapat melakukan:
        </p><p>✅ Prediksi komentar secara realtime<br>✅ Upload file CSV dan Excel<br>✅ Visualisasi hasil prediksi<br>✅ Analisis data komentar otomatis
        </p>
    </div>
    """, unsafe_allow_html=True)

    

    st.markdown("""
    <div class="info-box">
        <b>📢 Informasi:</b><br><br>

        Gunakan menu di sidebar untuk melakukan prediksi komentar,
        upload dataset, dan melihat evaluasi model.
    </div>
    """, unsafe_allow_html=True)

# ======================
# MENU PREDIKSI
# ======================
elif menu == "Prediksi":

    st.markdown("""
    <div class="card">
        <h2 style="color:#2563eb; text-align:center;">
            PREDIKSI KOMENTAR
        </h2>
    """, unsafe_allow_html=True)

    text = st.text_area(
        "Masukkan komentar TikTok:",
        height=150
    )

    if st.button("🚀 Prediksi Sekarang"):

        if text.strip() == "":
            st.warning("Masukkan komentar terlebih dahulu")

        else:

            try:

                pred = model.predict([text])[0]

                if str(pred).lower().strip() == "cyberbullying":

                    st.error("""
                    🚨 Komentar terdeteksi sebagai Cyberbullying
                    """)

                else:

                    st.success("""
                    ✅ Komentar terdeteksi sebagai Non Cyberbullying
                    """)

            except Exception as e:

                st.error(f"Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================
# MENU UPLOAD
# ======================
elif menu == "Upload CSV / XLSX":

    st.markdown("""
    <div class="card">
        <h2 style="color:#2563eb; text-align:center;">
            UPLOAD DATASET
        </h2>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload file CSV / XLSX",
        type=["csv", "xlsx", "xls"]
    )

    st.info("Gunakan kolom: komentar / comment / text / ulasan")

    if uploaded_file is not None:

        try:

            file_name = uploaded_file.name.lower()

            # ======================
            # CSV
            # ======================
            if file_name.endswith(".csv"):

                try:

                    df = pd.read_csv(
                        uploaded_file,
                        sep=',',
                        encoding='utf-8',
                        on_bad_lines='skip'
                    )

                except:

                    df = pd.read_csv(
                        uploaded_file,
                        sep=';',
                        encoding='latin1',
                        on_bad_lines='skip'
                    )

            # ======================
            # EXCEL
            # ======================
            elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):

                df = pd.read_excel(uploaded_file)

            else:

                st.error("Format file tidak didukung")
                st.stop()

            # ======================
            # NORMALISASI
            # ======================
            df.columns = df.columns.str.strip().str.lower()

            st.write("📌 Kolom ditemukan:", df.columns.tolist())

            # ======================
            # CARI KOLOM
            # ======================
            possible_cols = [
                'komentar',
                'comment',
                'text',
                'ulasan'
            ]

            found_col = None

            for col in possible_cols:

                if col in df.columns:
                    found_col = col
                    break

            # ======================
# PREDIKSI
# ======================
if found_col:

    # ======================
    # BERSIHKAN DATA
    # ======================
    df[found_col] = df[found_col].fillna("").astype(str)

    # hapus komentar kosong
    df = df[df[found_col].str.strip() != ""]

    # ======================
    # PREDIKSI
    # ======================
    df['prediksi'] = model.predict(
        df[found_col]
    )

    st.success("✅ Prediksi berhasil dilakukan")

    st.dataframe(df)
                # ======================
                # GRAFIK
                # ======================
                st.subheader("📊 Distribusi Prediksi")

                counts = df['prediksi'].value_counts()

                colors = []

                for label in counts.index:

                    label_clean = str(label).lower()

                    if "non" in label_clean:
                        colors.append("#22c55e")
                    else:
                        colors.append("#ef4444")

                fig, ax = plt.subplots(figsize=(6,4))

                counts.plot(
                    kind='bar',
                    ax=ax,
                    color=colors
                )

                plt.title("Distribusi Hasil Prediksi")
                plt.xlabel("Kelas")
                plt.ylabel("Jumlah")

                for i, v in enumerate(counts):

                    ax.text(
                        i,
                        v,
                        str(v),
                        ha='center',
                        va='bottom'
                    )

                st.pyplot(fig)

            else:

                st.error("❌ Kolom komentar tidak ditemukan")
                st.write("Kolom tersedia:", df.columns.tolist())

        except Exception as e:

            st.error(f"❌ Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================
# MENU EVALUASI
# ======================
elif menu == "Evaluasi Model":

    st.markdown("""
    <div class="card">
        <h2 style="color:#2563eb; text-align:center;">
            📈 EVALUASI MODEL
        </h2>
    """, unsafe_allow_html=True)

    labels = [
        "Cyberbullying",
        "Non Cyberbullying"
    ]

    values = [50, 50]

    fig2, ax2 = plt.subplots(figsize=(6,4))

    ax2.bar(
        labels,
        values,
        color=['#ef4444', '#22c55e']
    )

    plt.title("Simulasi Evaluasi Model")
    plt.ylabel("Jumlah")

    for i, v in enumerate(values):

        ax2.text(
            i,
            v,
            str(v),
            ha='center',
            va='bottom'
        )

    st.pyplot(fig2)

    st.markdown("""
    <div class="info-box">
        <b>📌 Keterangan:</b><br><br>

        Grafik di atas merupakan simulasi distribusi hasil evaluasi model
        klasifikasi cyberbullying menggunakan metode SVM.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
