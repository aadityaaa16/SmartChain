import pandas as pd
import joblib
import os
import optuna
import shap
import matplotlib.pyplot as plt
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("data/processed/supply_featured.csv")
print("Data loaded —", df.shape)


categorical_features = [
    "shipping_mode",
    "market",
    "order_region",
    "category"
]

numeric_features = [
    "quantity",
    "sales",
    "profit",
    "scheduled_shipping_days"
]

features = numeric_features + categorical_features
target = "risk_level"

df = df[features + [target]].dropna()

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")


preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


def objective(trial):

    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 3, 15)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])

    model.fit(X_train, y_train)

    return model.score(X_test, y_test)

print("\nRunning Optuna — 20 trials...")

study = optuna.create_study(direction="maximize")

study.optimize(
    objective,
    n_trials=20,
    show_progress_bar=True
)

print("\nBest settings :", study.best_params)
print("Best accuracy :", round(study.best_value * 100, 2), "%")


print("\nTraining final model...")

rf = RandomForestClassifier(
    **study.best_params,
    random_state=42
)

best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf)
])

best_model.fit(X_train, y_train)

# Evaluation
y_pred = best_model.predict(X_test)

print("\nModel Performance:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low Risk", "High Risk"]
    )
)


# SHAP Explainability

print("\nRunning SHAP explainability analysis...")

X_sample = X_test.iloc[:500]

X_transformed = best_model.named_steps["preprocessor"].transform(X_sample)

rf_model = best_model.named_steps["classifier"]

print(type(X_transformed))

if hasattr(X_transformed, "dtype"):
    print("dtype:", X_transformed.dtype)

try:
    print(X_transformed[:5])
except:
    pass

explainer = shap.TreeExplainer(rf_model)

# Convert sparse matrix to dense array — SHAP requires this
import scipy.sparse
if scipy.sparse.issparse(X_transformed):
    X_transformed = X_transformed.toarray()

shap_values = explainer.shap_values(X_transformed)

feature_names = best_model.named_steps[
    "preprocessor"
].get_feature_names_out()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

os.makedirs("src/models/saved", exist_ok=True)

plt.savefig("src/models/saved/risk_shap_summary.png")

plt.close()

print("SHAP summary plot saved.")


joblib.dump(
    best_model,
    "src/models/saved/delivery_risk_model.pkl"
)

print("Model saved to src/models/saved/delivery_risk_model.pkl")