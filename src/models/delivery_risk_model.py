import pandas as pd
import joblib
import os
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("data/processed/supply_featured.csv")
print("Data loaded —", df.shape)

df["shipping_mode"] = df["shipping_mode"].astype("category").cat.codes
df["market"]        = df["market"].astype("category").cat.codes
df["order_region"]  = df["order_region"].astype("category").cat.codes
df["category"]      = df["category"].astype("category").cat.codes
df["scheduled_shipping_days"] = df["scheduled_shipping_days"].astype("category").cat.codes


features = [
    "quantity",
    "sales",
    "profit",
    "shipping_mode",
    "market",
    "order_region",
    "category",
    "scheduled_shipping_days"
]

# 0 = low risk, 1 = high risk
target = "risk_level"


df = df[features + [target]].dropna()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")


#  Optuna finds best settings
# Each trial tries different settings and picks the best one
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth    = trial.suggest_int("max_depth", 3, 15)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


print("\nRunning Optuna — 50 trials...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nBest settings :", study.best_params)
print("Best accuracy :", round(study.best_value * 100, 2), "%")

print("\nTraining final model with best settings")
best_model = RandomForestClassifier(
    **study.best_params,
    random_state=42
)
best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_test)
print("\nModel Performance:")
print(classification_report(
    y_test, y_pred,
    target_names=["Low Risk", "High Risk"]
))

# SHAP explainability
# This explains WHY the model made each prediction —
# which features pushed risk score up or down
print("\nRunning SHAP explainability analysis...")

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test[:500])   # sample 500 rows for speed

shap.summary_plot(shap_values, X_test[:500], show=False)
plt.tight_layout()
plt.savefig("src/models/saved/risk_shap_summary.png")
plt.close()
print("SHAP summary plot saved to src/models/saved/delivery_risk_shap_summary.png")

os.makedirs("src/models/saved", exist_ok=True)
joblib.dump(best_model, "src/models/saved/delivery_risk_model.pkl")
print("Model saved to src/models/saved/delivery_risk_model.pkl")