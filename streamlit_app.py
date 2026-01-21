import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Log Anomaly Detection", layout="wide")
st.title("HDFS Log Anomaly Detection (On-the-fly ML Training)")

@st.cache_data
def load_data():
    # Load a small sample (you can replace this with Kaggle API later)
    df = pd.read_csv("parsed_hdfs_logs.csv", nrows=200000)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_resource
def build_model(df):
    texts = df["message"].astype(str)

    vectorizer = TfidfVectorizer(
        max_features=800,
        stop_words="english",
        ngram_range=(1,2),
        min_df=5
    )
    X_tfidf = vectorizer.fit_transform(texts)

    df["log_length"] = texts.str.len()
    df["word_count"] = texts.str.split().apply(len)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[["log_length", "word_count"]])

    X = hstack([X_tfidf, X_num])

    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        n_jobs=-1,
        random_state=42
    )

    labels = model.fit_predict(X)
    scores = model.decision_function(X)

    df["anomaly"] = (labels == -1).astype(int)
    df["score"] = scores

    return df, model

with st.spinner("Loading logs and training model (first run only)..."):
    data = load_data()
    data, model = build_model(data)

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Logs", len(data))
col2.metric("Anomalies Detected", int(data["anomaly"].sum()))
col3.metric("Anomaly Rate (%)", round(data["anomaly"].mean() * 100, 2))

st.divider()

# Timeline
st.subheader("Anomaly Timeline (per minute)")
timeline = data[data["anomaly"] == 1].set_index("timestamp").resample("1T").size()

fig1, ax1 = plt.subplots(figsize=(10,4))
timeline.plot(ax=ax1)
ax1.set_title("Anomaly Frequency Over Time")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# Score Distribution
st.subheader("Anomaly Score Distribution")
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.hist(data["score"], bins=50)
ax2.set_title("Isolation Forest Scores")
st.pyplot(fig2)

# Top Anomalies
st.subheader("Top 20 Most Anomalous Log Messages")
top_logs = data.sort_values("score").head(20)
st.dataframe(top_logs[["timestamp", "component", "message", "score"]])
