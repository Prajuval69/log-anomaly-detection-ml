import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.sparse import csr_matrix

# Load features
X = joblib.load("log_features.pkl")
X = X.tocsr()
# ---------------- Isolation Forest (Primary Model) ----------------
iso_forest = IsolationForest(
    n_estimators=200,
    max_samples=0.8,
    contamination=0.05,
    n_jobs=-1,
    random_state=42
)

iso_labels = iso_forest.fit_predict(X)
iso_scores = iso_forest.decision_function(X)
iso_anomaly = (iso_labels == -1).astype(int)

joblib.dump(iso_forest, "isolation_forest_model.pkl")

# ---------------- One-Class SVM (Trained on Subset) ----------------
SUBSET_SIZE = 120000
X_subset = X[:SUBSET_SIZE]

ocsvm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.03
)

ocsvm.fit(X_subset)

ocsvm_labels = ocsvm.predict(X)
ocsvm_scores = ocsvm.decision_function(X)
ocsvm_anomaly = (ocsvm_labels == -1).astype(int)

joblib.dump(ocsvm, "oneclass_svm_model.pkl")

# ---------------- Save Results ----------------
df_results = pd.DataFrame({
    "IsolationForest_Anomaly": iso_anomaly,
    "IsolationForest_Score": iso_scores,
    "OneClassSVM_Anomaly": ocsvm_anomaly,
    "OneClassSVM_Score": ocsvm_scores
})

df_results.to_csv("anomaly_labels.csv", index=False)

print("Anomaly Detection Models Trained Successfully")
print("Saved:")
print("- isolation_forest_model.pkl")
print("- oneclass_svm_model.pkl")
print("- anomaly_labels.csv with anomaly scores")
