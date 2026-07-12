import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Cyberbullying Detection",
    layout="wide"
)
# ======================
# LOAD DATASET UTAMA
# ======================

@st.cache_data
def load_data():

    data = pd.read_excel(
        "dataset_cyberbulyingfix.xlsx"
    )


    # rapikan nama kolom
    data.columns = (
        data.columns
        .str.lower()
        .str.strip()
    )


    # cari kolom komentar
    if "komentar" in data.columns:

        text_col = "komentar"

    elif "comment" in data.columns:

        text_col = "comment"

    elif "text" in data.columns:

        text_col = "text"

    elif "ulasan" in data.columns:

        text_col = "ulasan"

    else:

        raise Exception(
            "Kolom komentar tidak ditemukan"
        )



    # clean text
    data["clean_text"] = (
        data[text_col]
        .astype(str)
        .str.lower()
    )



    # buat label jika belum ada

    if "final_label" not in data.columns:


        data["binary_label"] = data["label"].map({

            "negatif":
            "cyberbullying",

            "netral":
            "non_cyberbullying",

            "positif":
            "non_cyberbullying"

        })


        data["final_label"] = (
            data["binary_label"]
        )


    return data




df = load_data()



# ======================
# LOAD MODEL
# ======================

@st.cache_resource
def load_model():

    return joblib.load(
        "MODEL_BARU.pkl"
    )



model = load_model()

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
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # MAPPING LABEL
    # ==========================================
    def hasil_label(pred):
        pred = str(pred).lower().strip()

        if pred == "cyberbullying":
            return "Cyberbullying"

        elif pred == "non_cyberbullying":
            return "Non Cyberbullying"

        return pred


    # ==========================================
    # KAMUS KATA CYBERBULLYING
    # ==========================================

    kata_penghinaan = [

    # Fisik
    "jelek",
    "jelek banget",
    "jelek amat",
    "muka jelek",
    "muka buruk",
    "muka hancur",
    "muka ancur",
    "muka serem",
    "muka rusak",
    "muka kayak monyet",
    "muka kayak babi",
    "muka kayak setan",
    "muka kayak alien",
    "muka kayak tuyul",
    "muka kayak zombie",
    "muka jelek banget",
    "rambutnya keriting kayak mie",
    "cantik tapi oon"

    "kurus",
    "cantik tapi kurus",
    "kurus banget",
    "kurus kering",
    "kurus kayak tiang",
    "kurus kayak lidi",
    "kurus kayak kerangka",

    "gendut",
    "gendut banget",
    "gendut kayak babi",
    "gendut kayak sapi",
    "gendut parah",
    "obesitas",

    "item",
    "hitam",
    "hitam banget",
    "gosong",
    "dekil",
    "jorok",
    "bau",
    "bau badan",
    "bau mulut",

    "pesek",
    "hidung pesek",
    "mata sipit",
    "bibir tebal",
    "kuping besar",
    "kepala botak",
    "botak",
    "rambut keriting",
    "kayak tiang"

    # Intelegensi
    "bodoh",
    "tolol",
    "goblok",
    "bloon",
    "bego",
    "dungu",
    "idiot",
    "otak udang",
    "otak kosong",
    "otak jongkok",
    "gak punya otak",
    "kurang otak",
    "otaknya dimana",

    # Status Sosial
    "miskin",
    "kere",
    "pengemis",
    "gelandangan",
    "kampungan",
    "norak",
    "alay",
    "sok kaya",
    "sok miskin",
    "murahan",
    "tidak berguna",
    "gak berguna",
    "pecundang",
    "beban",
    "parasit",
    "sampah masyarakat",

    # Hewan
    "anjing",
    "asu",
    "babi",
    "monyet",
    "kera",
    "keledai",
    "anjir",
    "kampret",
    "tai",
    "setan",
    "iblis",

    # Kata Kasar
    "bangsat",
    "brengsek",
    "kontol",
    "memek",
    "ngentot",
    "tai",
    "tai kucing",
    "bajingan",
    "keparat",
    "laknat",
    "tolol banget",
    "goblok banget",
    "bodoh banget",
    "idiot banget",
    "tolol lu",
    "goblok lu",
    "dasar goblok",
    "dasar tolol",
    "dasar bodoh",
    "dasar bego",
    "dasar idiot",
    "dasar bangsat",
    "dasar kampungan",
    "dasar miskin",
    "dasar jelek",
    "dasar sampah",
    "dasar pecundang",

    # Ancaman
    "mati aja",
    "mati sana",
    "cepat mati",
    "mending mati",
    "mati kau",
    "mati lo",
    "bunuh diri",
    "gantung diri",
    "lompat aja",
    "hilang aja",
    "musnah aja",

    # Penghinaan Umum
    "najis",
    "jijik",
    "memalukan",
    "malu maluin",
    "aib",
    "hina",
    "ampas",
    "brengsek lu",
    "gak ada guna",
    "gak ada manfaat",
    "manusia sampah",
    "orang sampah",
    "manusia gagal",
    "gagal total",
    "pecundang sejati",
    "tidak pantas hidup",
    "tidak layak hidup",

    # Cyberbullying Modern
    "cancel aja",
    "unfollow aja",
    "laporin aja",
    "block aja",
    "blok aja",
    "ga usah hidup",
    "gak usah hidup",
    "mending keluar",
    "gak ada yang suka",
    "semua benci kamu",
    "semua jijik sama kamu",
    "semua muak sama kamu",
    "orang paling jelek",
    "orang paling bodoh",
    "orang paling tolol",

    # Frasa
    "kamu jelek",
    "kamu bodoh",
    "kamu goblok",
    "kamu tolol",
    "kamu idiot",
    "kamu kampungan",
    "kamu miskin",
    "kamu bego",
    "kamu sampah",
    "kamu pecundang",
    "lu jelek",
    "lu goblok",
    "lu tolol",
    "lu bego",
    "lu idiot",
    "lu miskin",
    "lu kampungan",
    "lu sampah",
    "kau bodoh",
    "kau goblok",
    "kau jelek",
    "kau tolol",
    "dia jelek",
    "dia goblok",
    "dia tolol",
    "dia bodoh",
    "orang ini goblok",
    "orang ini jelek",
    "orang ini sampah",
    "orang ini bodoh"
]


    kata_target = [

    # Kata ganti orang kedua
    "kamu",
    "kau",
    "anda",
    "dirimu",
    "diri kamu",
    "diri lu",
    "diri lo",
    "engkau",

    # Bahasa gaul
    "lu",
    "loe",
    "lo",
    "elu",
    "elo",
    "luu",
    "loe",
    "lw",
    "luw",
    "eloe",

    # Orang ketiga
    "dia",
    "doi",
    "beliau",
    "orang itu",
    "orang ini",
    "nih orang",
    "si dia",
    "si itu",
    "si ini",

    # Panggilan
    "bro",
    "brok",
    "broh",
    "bray",
    "bang",
    "abang",
    "mas",
    "mba",
    "mbak",
    "kak",
    "kaka",
    "kakak",
    "dek",
    "adik",
    "bos",
    "boss",
    "bung",
    "cuy",
    "cuyy",
    "cok",
    "cuk",
    "rek",

    # Sebutan orang
    "orang",
    "orangnya",
    "cowok",
    "cewek",
    "laki",
    "laki laki",
    "perempuan",
    "pria",
    "wanita",
    "anak",
    "bocah",
    "bocil",

    # Bagian tubuh yang sering menjadi target hinaan
    "muka",
    "wajah",
    "mukamu",
    "wajahmu",
    "badan",
    "tubuh",
    "kepala",
    "hidung",
    "mata",
    "kuping",
    "telinga",
    "gigi",
    "rambut",
    "kulit",

    # Bentuk kepemilikan
    "kamu itu",
    "kau itu",
    "lu itu",
    "lo itu",
    "dia itu",
    "orang ini",
    "orang itu",
    "muka kamu",
    "muka lu",
    "muka lo",
    "badan kamu",
    "badan lu",
    "badan lo",
    "otak kamu",
    "otak lu",
    "otak lo",
    "hidung kamu",
    "wajah kamu",
    "kulit kamu",

    # Sebutan umum di media sosial
    "admin",
    "creator",
    "konten kreator",
    "youtuber",
    "tiktoker",
    "streamer",
    "influencer",
    "artis",
    "seleb",
    "selebgram",
    "idol"
]


    text = st.text_area(
        "Masukkan komentar TikTok:",
        height=150,
        placeholder="Contoh: Kamu jelek banget"
    )


    if st.button("🚀 Prediksi Sekarang"):

        if text.strip() == "":

            st.warning("⚠️ Masukkan komentar terlebih dahulu")

        else:

            try:

                komentar = text.lower()

                # ==========================================
                # RULE BASED
                # ==========================================

                ada_penghinaan = any(
                    kata in komentar
                    for kata in kata_penghinaan
                )

                ada_target = any(
                    kata in komentar
                    for kata in kata_target
                )


                # ==========================================
                # PRIORITAS RULE
                # ==========================================

                if ada_penghinaan and ada_target:

                    hasil = "cyberbullying"

                else:

                    hasil = model.predict([text])[0]


                label = hasil_label(hasil)


                # ==========================================
                # HASIL
                # ==========================================

                if label == "Cyberbullying":

                    st.markdown(f"""
                    <div style="
                    background:#fee2e2;
                    padding:25px;
                    border-radius:15px;
                    color:#dc2626;
                    font-size:20px;">

                    🚨 <b>Cyberbullying</b>

                    <br><br>

                    Komentar yang dianalisis:

                    <br>

                    <i>"{text}"</i>

                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown(f"""
                    <div style="
                    background:#dcfce7;
                    padding:25px;
                    border-radius:15px;
                    color:#15803d;
                    font-size:20px;">

                    ✅ <b>Non Cyberbullying</b>

                    <br><br>

                    Komentar yang dianalisis:

                    <br>

                    <i>"{text}"</i>

                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:

                st.error(f"Terjadi error pada model: {e}")

# ======================
# MENU UPLOAD
# ======================
elif menu == "Upload CSV / XLSX":
    st.markdown("""
    <div class="card">
        <h2 style="color:#2563eb; text-align:center;">
            ANALISIS DATASET CYBERBULLYING
        </h2>
    </div>
    """, unsafe_allow_html=True)



    uploaded_file = st.file_uploader(
        "Upload file CSV / XLSX",
        type=["csv","xlsx","xls"]
    )


    st.info(
        "Kolom komentar yang diterima: komentar / comment / text / ulasan"
    )



    if uploaded_file is not None:


        try:


            # ======================
            # BACA FILE
            # ======================

            file_name = uploaded_file.name.lower()


            if file_name.endswith(".csv"):


                try:

                    df = pd.read_csv(
                        uploaded_file,
                        encoding="utf-8",
                        on_bad_lines="skip"
                    )

                except:

                    df = pd.read_csv(
                        uploaded_file,
                        encoding="latin1",
                        sep=";",
                        on_bad_lines="skip"
                    )


            else:


                df = pd.read_excel(
                    uploaded_file
                )




            # ======================
            # NORMALISASI
            # ======================

            df.columns = (
                df.columns
                .str.lower()
                .str.strip()
            )


            st.write(
                "Kolom:",
                df.columns.tolist()
            )



            cari_kolom = [
                "komentar",
                "comment",
                "text",
                "ulasan"
            ]


            found_col = None


            for col in cari_kolom:

                if col in df.columns:

                    found_col = col
                    break



            if found_col is None:


                st.error(
                    "Kolom komentar tidak ditemukan"
                )

                st.stop()



            # ======================
            # PREDIKSI SVM
            # ======================

            df[found_col] = (
                df[found_col]
                .astype(str)
            )


            df["prediksi"] = model.predict(
                df[found_col]
            )



            # ======================
            # PROBABILITAS SVM
            # ======================

            if hasattr(
                model,
                "predict_proba"
            ):


                probability = model.predict_proba(
                    df[found_col]
                )


                classes = model.classes_


                index_cyber = list(classes).index(
                    "cyberbullying"
                )



                df["Cyberbullying (%)"] = (
                    probability[:,index_cyber] * 100
                ).round(2)



                df["Non Cyberbullying (%)"] = (
                    100 -
                    df["Cyberbullying (%)"]
                ).round(2)


            else:


                df["Cyberbullying (%)"] = 0

                df["Non Cyberbullying (%)"] = 0




            st.success(
                "✅ Analisis berhasil"
            )



            # ======================
            # TABEL HASIL
            # ======================

            st.markdown(
                "## HASIL PREDIKSI"
            )


            st.dataframe(
                df,
                use_container_width=True
            )




            # ======================
            # PERHITUNGAN
            # ======================

            total = len(df)


            cyber = (
                df["prediksi"]
                .astype(str)
                .str.lower()
                .eq("cyberbullying")
                .sum()
            )


            non = total - cyber



            cyber_percent = (
                cyber / total * 100
            )


            non_percent = (
                non / total * 100
            )




            # ======================
            # CARD DASHBOARD
            # ======================


            st.markdown(
                "## RINGKASAN ANALISIS"
            )



            col1,col2,col3 = st.columns(3)



            with col1:

                st.markdown(f"""
                <div style="
                background:#fee2e2;
                padding:25px;
                border-radius:20px;
                text-align:center;">

                <h1></h1>

                <h2>{cyber_percent:.2f}%</h2>

                <b>Cyberbullying</b>

                </div>
                """,
                unsafe_allow_html=True)



            with col2:


                st.markdown(f"""
                <div style="
                background:#dcfce7;
                padding:25px;
                border-radius:20px;
                text-align:center;">

                <h1></h1>

                <h2>{non_percent:.2f}%</h2>

                <b>Non Cyberbullying</b>

                </div>
                """,
                unsafe_allow_html=True)




            with col3:


                st.markdown(f"""
                <div style="
                background:#dbeafe;
                padding:25px;
                border-radius:20px;
                text-align:center;">

                <h1></h1>

                <h2>{total}</h2>

                <b>Total Data</b>

                </div>
                """,
                unsafe_allow_html=True)




            # ======================
            # LEVEL RISIKO
            # ======================


            st.markdown(
                "## ⚠️ Tingkat Risiko"
            )



            if cyber_percent >= 70:

                risiko = "TINGGI 🚨"

            elif cyber_percent >= 40:

                risiko = "SEDANG ⚠️"

            else:

                risiko = "RENDAH ✅"



            st.info(
                f"Tingkat Cyberbullying: {risiko}"
            )



            st.progress(
                int(cyber_percent)/100
            )




            # ======================
            # GRAFIK
            # ======================

            st.markdown(
                "## GRAFIK PRESENTASE"
            )


            grafik = pd.DataFrame({

                "Kategori":[
                    "Cyberbullying",
                    "Non Cyberbullying"
                ],

                "Persentase":[
                    cyber_percent,
                    non_percent
                ]

            })



            fig,ax = plt.subplots(
                figsize=(7,4)
            )


            ax.bar(
                grafik["Kategori"],
                grafik["Persentase"]
            )


            ax.set_ylim(
                0,
                100
            )


            ax.set_ylabel(
                "Persentase (%)"
            )


            ax.set_title(
                "Deteksi Cyberbullying Menggunakan SVM"
            )


            for i,v in enumerate(
                grafik["Persentase"]
            ):

                ax.text(
                    i,
                    v,
                    f"{v:.2f}%",
                    ha="center"
                )


            st.pyplot(fig)




            # ======================
            # INFORMASI MODEL
            # ======================

            with st.expander(
                "🤖 Detail Model"
            ):

                st.write(
                    """
                    Algoritma:
                    Support Vector Machine (SVM)

                    Fitur:
                    TF-IDF Vectorizer

                    Optimasi:
                    Class Weight

                    Output:
                    Probabilitas tingkat Cyberbullying (%)
                    """
                )



        except Exception as e:


            st.error(
                f"Terjadi error: {e}"
            )

# ======================
# MENU EVALUASI MODEL
# ======================
elif menu == "Evaluasi Model":


    st.markdown("""
    <h1 style="
    text-align:center;
    color:#2563eb;">
    📈 EVALUASI MODEL SVM
    </h1>
    """,
    unsafe_allow_html=True)



    try:


        from sklearn.model_selection import train_test_split

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report
        )



        # ======================
        # DATA EVALUASI
        # ======================

        X = df["clean_text"]

        y = df["final_label"]



        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y

        )



        # ======================
        # PREDIKSI MODEL
        # ======================

        pred = model.predict(
            X_test
        )



        # ======================
        # HITUNG METRIK
        # ======================

        accuracy = accuracy_score(
            y_test,
            pred
        )


        precision = precision_score(
            y_test,
            pred,
            average="weighted"
        )


        recall = recall_score(
            y_test,
            pred,
            average="weighted"
        )


        f1 = f1_score(
            y_test,
            pred,
            average="weighted"
        )




        # ======================
        # CARD NILAI
        # ======================

        st.markdown(
            "## PERFORMA MODEL"
        )


        col1,col2,col3,col4 = st.columns(4)


        col1.metric(
            "Accuracy",
            f"{accuracy*100:.2f}%"
        )


        col2.metric(
            "Precision",
            f"{precision*100:.2f}%"
        )


        col3.metric(
            "Recall",
            f"{recall*100:.2f}%"
        )


        col4.metric(
            "F1 Score",
            f"{f1*100:.2f}%"
        )





        # ======================
        # GRAFIK METRIK
        # ======================

        st.markdown(
            "## GRAFIK EVALUASI"
        )


        evaluasi = pd.DataFrame({

            "Metrik":[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],


            "Nilai":[
                accuracy*100,
                precision*100,
                recall*100,
                f1*100
            ]

        })



        fig,ax = plt.subplots(
            figsize=(7,4)
        )


        ax.bar(
            evaluasi["Metrik"],
            evaluasi["Nilai"]
        )


        ax.set_ylim(
            0,
            100
        )


        ax.set_ylabel(
            "Persentase (%)"
        )


        ax.set_title(
            "Evaluasi Performa SVM + TF-IDF + Class Weight"
        )



        for i,v in enumerate(
            evaluasi["Nilai"]
        ):

            ax.text(
                i,
                v,
                f"{v:.2f}%",
                ha="center"
            )



        st.pyplot(fig)





        # ======================
        # CONFUSION MATRIX
        # ======================

        st.markdown(
            "## CONFISION MATRIX"
        )


        cm = confusion_matrix(
            y_test,
            pred
        )


        fig2,ax2 = plt.subplots(
            figsize=(5,4)
        )


        ax2.imshow(
            cm
        )


        ax2.set_xticks(
            [0,1]
        )


        ax2.set_yticks(
            [0,1]
        )


        ax2.set_xticklabels(
            [
                "Cyberbullying",
                "Non Cyberbullying"
            ]
        )


        ax2.set_yticklabels(
            [
                "Cyberbullying",
                "Non Cyberbullying"
            ]
        )


        ax2.set_xlabel(
            "Prediksi"
        )


        ax2.set_ylabel(
            "Aktual"
        )


        for i in range(2):

            for j in range(2):

                ax2.text(
                    j,
                    i,
                    cm[i,j],
                    ha="center",
                    va="center"
                )


        st.pyplot(fig2)





        # ======================
        # REPORT
        # ======================

        with st.expander(
            "Classification Report"
        ):

            st.text(
                classification_report(
                    y_test,
                    pred
                )
            )




        st.success(
            """
            Evaluasi berhasil dilakukan menggunakan:
            
            ✔ Support Vector Machine (SVM)
            
            ✔ TF-IDF Vectorizer
            
            ✔ Class Weight
            
            ✔ Data testing 20%
            """
        )



    except Exception as e:


        st.error(
            f"Gagal melakukan evaluasi: {e}"
        )
