import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib

# Load representative sample for scalability
SAMPLE_SIZE = 400000
df = pd.read_csv("parsed_hdfs_logs.csv", nrows=SAMPLE_SIZE)

texts = df["message"].astype(str)

# TF-IDF with pruning for speed + quality
vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=5,
    max_df=0.8,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_tfidf = vectorizer.fit_transform(texts)

# Lightweight numeric features
df["log_length"] = texts.str.len()
df["word_count"] = texts.str.split().apply(len)

numeric_features = df[["log_length", "word_count"]]
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)

# Combine sparse + dense
X_final = hstack([X_tfidf, numeric_scaled])

# Save artifacts
joblib.dump(X_final, "log_features.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "numeric_scaler.pkl")

print("Feature Engineering Completed")
print("Samples used:", SAMPLE_SIZE)
print("Final Feature Shape:", X_final.shape)
