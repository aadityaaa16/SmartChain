import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/processed/supply_featured.csv")
print("Data loaded —", df.shape)


features = [
    "quantity",
    "sales",
    "profit",
    "delay_days",
    "is_late",
    "actual_shipping_days",
    "scheduled_shipping_days"
]

df = df[features].dropna()
print(f"Rows used for anomaly detection: {len(df)}")

# Train Isolation Forest with sensible default settings
print("\nTraining anomaly detection model...")
model = IsolationForest(
    n_estimators=150,
    contamination=0.05,   # assume ~5% of orders are unusual
    random_state=42
)
model.fit(df)


# -1 = anomaly, 1 = normal
df["anomaly"] = model.predict(df[features])
df["anomaly"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})


#  Store anomaly scores 

df["anomaly_score"] = model.decision_function(df[features])

print("\nAnomaly detection results:")
print(df["anomaly"].value_counts())

# Show most severe anomalies first
print("\nTop 5 most severe anomalies (lowest score = most unusual):")
print(
    df[df["anomaly"] == "Anomaly"]
    .sort_values("anomaly_score")
    .head(5)
)

# SHAP explainability
print("\nRunning SHAP explainability analysis...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df[features][:500])

shap.summary_plot(shap_values, df[features][:500], show=False)
plt.tight_layout()
plt.savefig("src/models/saved/anomaly_shap_summary.png")
plt.close()
print("SHAP summary plot saved to src/models/saved/anomaly_shap_summary.png")

#  Save model and scored data 
os.makedirs("src/models/saved", exist_ok=True)
joblib.dump(model, "src/models/saved/anomaly_model.pkl")

df.to_csv("data/processed/anomalies_scored.csv", index=False)

print("\nModel saved to src/models/saved/anomaly_model.pkl")
print("Scored data saved to data/processed/anomalies_scored.csv")