import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load parsed logs and anomaly results
logs = pd.read_csv("parsed_hdfs_logs.csv", nrows=400000)
anomalies = pd.read_csv("anomaly_labels.csv")

# Merge
data = pd.concat([logs.reset_index(drop=True), anomalies], axis=1)

# Convert timestamp
data["timestamp"] = pd.to_datetime(data["timestamp"])

# -------------------------------
# 1. Anomaly Score Distribution
# -------------------------------
plt.figure(figsize=(8,4))
sns.histplot(data["IsolationForest_Score"], bins=50, kde=True)
plt.title("Isolation Forest Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 2. Anomaly Timeline
# -------------------------------
anomaly_time = data[data["IsolationForest_Anomaly"] == 1].set_index("timestamp")

plt.figure(figsize=(10,4))
anomaly_time.resample("1T").size().plot()
plt.title("Anomaly Timeline (per minute)")
plt.xlabel("Time")
plt.ylabel("Number of Anomalies")
plt.show()

# -------------------------------
# 3. Top Anomalous Log Messages
# -------------------------------
top_anomalies = data.sort_values("IsolationForest_Score").head(20)

print("\nTop 20 Most Anomalous Log Entries:\n")
for i, row in top_anomalies.iterrows():
    print(f"[{row['timestamp']}] {row['message']}")
