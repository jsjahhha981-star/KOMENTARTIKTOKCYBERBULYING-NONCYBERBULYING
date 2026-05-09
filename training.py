import pandas as pd
import re
import joblib

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ======================
# LOAD DATA
# ======================
df = pd.read_excel("dataset_cyberbulyingfix.xlsx")

# ======================
# PREPROCESS
# ======================
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = stopword.remove(text)
    text = stemmer.stem(text)

    return text

df["text_bersih"] = df["komentar"].apply(clean_text)

# ======================
# LABEL
# ======================
df["label_final"] = df["label"].map({
    "negatif": "cyberbullying",
    "netral": "non_cyberbullying",
    "positif": "non_cyberbullying"
})

# ======================
# DATA
# ======================
X = df["text_bersih"].astype(str)
y = df["label_final"]

# ======================
# SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================
# PIPELINE
# ======================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("svm", SVC(kernel="linear", class_weight="balanced"))
])

# ======================
# TRAIN
# ======================
pipeline.fit(X_train, y_train)

# ======================
# TEST
# ======================
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# ======================
# VALIDASI TFIDF
# ======================
print(pipeline)

print(
    "TFIDF READY:",
    hasattr(pipeline.named_steps['tfidf'], 'idf_')
)

# ======================
# SAVE MODEL
# ======================
joblib.dump(pipeline, "MODEL_AMAN.pkl")

print("MODEL BERHASIL DISIMPAN")