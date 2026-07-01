from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

# Global model storage
# Models are loaded once when the server starts
# not on every request
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models at startup
    print("Loading models...")

    # Delivery risk model
    models["risk"] = joblib.load("src/models/saved/delivery_risk_model.pkl")
    print("Risk model loaded")

    # Anomaly detection model
    models["anomaly"] = joblib.load("src/models/saved/anomaly_model.pkl")
    print("Anomaly model loaded")

    # LSTM demand model
    models["lstm"] = tf.keras.models.load_model("src/models/saved/best_lstm_model.keras")
    models["lstm_scaler"] = joblib.load("src/models/saved/lstm_scaler.pkl")
    with open("src/models/saved/lstm_features.json") as f:
        models["lstm_features"] = json.load(f)
    print("LSTM model loaded")

    # Supplier risk scores (pre-computed)
    models["supplier_risk"] = pd.read_csv("data/processed/supplier_risk_scores.csv")
    print("Supplier risk scores loaded")

    # FinBERT sentiment model
    print("Loading FinBERT — this takes 20-30 seconds...")
    models["sentiment"] = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        framework="tf"
    )
    print("FinBERT loaded")

    print("\nAll models ready!")
    yield

    
    models.clear()


# Create FastAPI app 
app = FastAPI(
    title="SmartChain API",
    description="AI-powered supply chain intelligence platform",
    version="1.0.0",
    lifespan=lifespan
)

# Allow Streamlit to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


#  Health check
@app.get("/")
def home():
    return {"status": "SmartChain API is running"}


# ENDPOINT 1 — Delivery Risk

@app.post("/predict-risk")
def predict_risk(data: dict):
    """
    Predicts whether a delivery order is low or high risk.
    Input: order features as a dictionary
    """
    try:
        # Convert text columns to category codes
        # Must match exactly what was done during training
        df = pd.DataFrame([data])

        for col in ["shipping_mode", "market", "order_region", "category"]:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes

        features = [
            "quantity", "sales", "profit",
            "shipping_mode", "market",
            "order_region", "category",
            "scheduled_shipping_days"
        ]

        X = df[features].values
        prediction = models["risk"].predict(X)[0]
        probability = models["risk"].predict_proba(X)[0].max()

        return {
            "risk_level": int(prediction),
            "risk_label": "High Risk" if prediction == 1 else "Low Risk",
            "confidence": round(float(probability), 3)
        }

    except Exception as e:
        return {"error": str(e)}



# ENDPOINT 2 — Anomaly Detection

@app.post("/detect-anomaly")
def detect_anomaly(data: dict):
    """
    Detects if an order has unusual patterns.
    Input: order features as a dictionary
    """
    try:
        df = pd.DataFrame([data])
        features = ["quantity", "sales", "profit",
                    "delay_days", "is_late",
                    "actual_shipping_days", "scheduled_shipping_days"]

        X = df[features].values
        prediction = models["anomaly"].predict(X)[0]
        score = models["anomaly"].decision_function(X)[0]

        return {
            "is_anomaly": bool(prediction == -1),
            "label": "Anomaly" if prediction == -1 else "Normal",
            "anomaly_score": round(float(score), 4)
        }

    except Exception as e:
        return {"error": str(e)}


# ENDPOINT 3 — Demand Forecast

@app.post("/forecast-demand")
def forecast_demand(data: dict):
    """
    Predicts next day demand given last 30 days of sales data.
    Input: last_30_days (list of 30 daily sales values)
    """
    try:
        last_30 = data.get("last_30_days", [])

        if len(last_30) != 30:
            return {"error": "Please provide exactly 30 days of sales data"}

        # Build feature array
        sales = np.array(last_30)
        month = data.get("month", 6)
        day_of_week = data.get("day_of_week", 0)
        day_of_year = data.get("day_of_year", 180)
        lag_7 = sales[-7]
        lag_14 = sales[-14]
        rolling_mean_7 = sales[-7:].mean()

       
        features = np.array([[
            sales[-1],
            month,
            day_of_week,
            day_of_year,
            lag_7,
            lag_14,
            rolling_mean_7
        ]])

        # Scale using saved scaler
        scaled = models["lstm_scaler"].transform(features)

        # Reshape for LSTM: (1, 1, n_features)
        X = scaled.reshape(1, 1, scaled.shape[1])

        # Predict
        prediction_scaled = models["lstm"].predict(X, verbose=0)

        # Inverse transform
        dummy = np.zeros((1, len(models["lstm_features"])))
        dummy[0, 0] = prediction_scaled[0, 0]
        predicted_sales = models["lstm_scaler"].inverse_transform(dummy)[0, 0]

        return {
            "predicted_demand": round(float(predicted_sales), 2),
            "unit": "daily sales units"
        }

    except Exception as e:
        return {"error": str(e)}



# ENDPOINT 4 — Supplier Risk

@app.get("/supplier-risk")
def get_supplier_risk():
    """
    Returns the pre-computed risk scores for all suppliers.
    """
    try:
        data = models["supplier_risk"][[
            "Supplier", "risk_score",
            "risk_category", "avg_delay_days",
            "avg_defect_rate", "compliance_rate"
        ]].to_dict(orient="records")

        return {"suppliers": data}

    except Exception as e:
        return {"error": str(e)}


@app.get("/supplier-risk/{supplier_name}")
def get_single_supplier_risk(supplier_name: str):
    """
    Returns risk score for a specific supplier.
    """
    try:
        df = models["supplier_risk"]
        supplier = df[df["Supplier"].str.lower() == supplier_name.lower()]

        if supplier.empty:
            return {"error": f"Supplier '{supplier_name}' not found"}

        return supplier.to_dict(orient="records")[0]

    except Exception as e:
        return {"error": str(e)}



# ENDPOINT 5 — Sentiment Analysis

@app.get("/sentiment/{company_name}")
def get_sentiment(company_name: str):
    """
    Fetches real news headlines for a company and
    runs FinBERT sentiment analysis on them.
    """
    try:
        NEWS_API_KEY = os.getenv("NEWS_API_KEY")

        # Fetch news
        url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        data = response.json()

        if data.get("status") != "ok":
            return {"error": data.get("message")}

        articles = data.get("articles", [])
        if not articles:
            return {"error": f"No news found for {company_name}"}

        # Run FinBERT on each headline
        results = []
        for article in articles:
            headline = article.get("title", "")
            if not headline:
                continue
            sentiment = models["sentiment"](headline)[0]
            results.append({
                "headline": headline,
                "sentiment": sentiment["label"],
                "confidence": round(sentiment["score"], 3),
                "published": article.get("publishedAt", "")
            })

        # Overall sentiment summary
        labels = [r["sentiment"] for r in results]
        overall = max(set(labels), key=labels.count)

        return {
            "company": company_name,
            "overall_sentiment": overall,
            "total_articles": len(results),
            "results": results
        }

    except Exception as e:
        return {"error": str(e)}