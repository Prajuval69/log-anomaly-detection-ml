# Log Anomaly Detection in HDFS using Unsupervised Machine Learning

This project implements an end-to-end anomaly detection system for large-scale server logs (HDFS / distributed systems) using unsupervised machine learning techniques.

The goal is to automatically identify abnormal log patterns that may indicate failures, security issues, or unexpected system behavior.

---

## 🚀 Key Features

- Log parsing and preprocessing  
- Text feature extraction using TF-IDF  
- Statistical feature engineering (message length, token counts)  
- Unsupervised anomaly detection using:
  - Isolation Forest  
  - (Optional) One-Class SVM  
- Anomaly scoring and ranking  
- Interactive visualization using Streamlit:
  - Anomaly timeline  
  - Score distribution  
  - Top anomalous log events  

---

## 🧠 Models Used

- Isolation Forest (primary model)  
- One-Class SVM (boundary-based anomaly detection)  
- TF-IDF Vectorization for log message representation  

---

## 🖥️ Streamlit Dashboard

The Streamlit app provides:

- Real-time anomaly statistics  
- Temporal visualization of anomaly spikes  
- Ranked anomalous log messages  
- Fully reproducible ML pipeline  

---

## ⚠️ Important Note on Reproducibility

### Why large datasets and trained models are NOT stored in this GitHub repo

The original HDFS log files and trained feature matrices/models are several gigabytes in size and exceed GitHub’s file size limits.

Instead of committing large artifacts, the project follows **industry-standard MLOps practice**:

- Models are trained dynamically at runtime inside the Streamlit app  
- Feature engineering and Isolation Forest training are executed on-the-fly  
- Heavy computations are cached using Streamlit’s resource caching  

### Result

This means:

- The GitHub / Streamlit Cloud version re-trains the model each time the cache is cleared  
- Anomaly scores and exact detections may slightly vary between runs  
- This behavior is expected and correct for unsupervised anomaly detection systems  
- The logical pipeline, methodology, and visual insights remain identical to the full offline version  

In real production systems, large models and datasets are stored in:

- Object storage (S3, GCS, HDFS, Artifact Stores)  
- Model registries (MLflow, HuggingFace Hub, internal MLOps platforms)  

They are not committed directly to Git repositories.

This repository intentionally demonstrates:

> A scalable, reproducible, cloud-deployable ML pipeline rather than static model files.

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- TF-IDF Vectorization  
- Isolation Forest, One-Class SVM  
- Streamlit  
- Matplotlib  

---

## 📊 Use Cases

- Data center monitoring  
- Cyber-security log inspection  
- Fault detection in distributed systems  
- AIOps and MLOps research  
- Proactive system health monitoring  

---

## 📄 Live Demo
https://log-anomaly.streamlit.app/

