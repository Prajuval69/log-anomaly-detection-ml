import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

st.set_page_config(page_title="Log Anomaly Detection", layout="wide")
st.title("HDFS Log Anomaly Detection using Unsupervised ML")

# ---------------------------
# 1. Generate Synthetic Logs
# ---------------------------
@st.cache_data
def generate_logs(n=20000):
    normal_msgs = [
        "Receiving block from DataNode",
        "Block allocation successful",
        "Heartbeat received from node",
        "PacketResponder completed",
        "Metadata update completed",
        "Replication process finished"
    ]

    anomaly_msgs = [
        "Connection timeout to DataNode",
        "Block corrupted during transfer",
        "Disk I/O error detected",
        "NameNode not responding",
        "Checksum verification failed",
        "Unexpected shutdown of DataNode"
    ]

    logs = []
    timestamps = pd.date_range("2025-01-01", periods=n, freq="min")

    for i in range(n):
        if np.random.rand() < 0.05:  # 5% anomalies
            msg = np.random.choice(anomaly_msgs)
        else:
            msg = np.random.choice(normal_msgs)

        logs.append(msg)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "message": logs
    })

    return df

# ---------------------------
# 2. Train Model On-the-Fly
# ---------------------------
@st.cache_resource
def train_model(df):
    texts = df["message"]

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2))
    X_text = vectorizer.fit_transform(texts)

    df["length"] = texts.str.len()
    df["words"] = texts.str.split().apply(len)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[["length", "words"]])

    X = hstack([X_text, X_num])

    model = IsolationForest(contamination=0.05, n_estimators=150, random_state=42)
    labels = model.fit_predict(X)
    scores = model.decision_function(X)

    df["anomaly"] = (labels == -1).astype(int)
    df["score"] = scores

    return df, model

with st.spinner("Generating logs and training Isolation Forest (first run only)..."):
    data = generate_logs()
    data, model = train_model(data)

# ---------------------------
# 3. Dashboard
# ---------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Logs", len(data))
col2.metric("Detected Anomalies", int(data["anomaly"].sum()))
col3.metric("Anomaly Rate (%)", round(data["anomaly"].mean() * 100, 2))

st.divider()

# Timeline
st.subheader("Anomaly Timeline")
timeline = data[data["anomaly"] == 1].set_index("timestamp").resample("1H").size()

fig1, ax1 = plt.subplots(figsize=(10,4))
timeline.plot(ax=ax1)
ax1.set_ylabel("Anomalies per Hour")
st.pyplot(fig1)

# Score distribution
st.subheader("Anomaly Score Distribution")
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.hist(data["score"], bins=50)
ax2.set_xlabel("Isolation Forest Score")
st.pyplot(fig2)

# Top anomalies
st.subheader("Top 20 Anomalous Log Events")
top = data.sort_values("score").head(20)
st.dataframe(top[["timestamp", "message", "score"]])

st.caption("Unsupervised Anomaly Detection using TF-IDF + Isolation Forest | Fully Reproducible | Cloud Deployable")
