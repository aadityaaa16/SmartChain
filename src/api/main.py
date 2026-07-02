from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
from dotenv import load_dotenv
import os

load_dotenv()

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")

    models["risk"] = joblib.load("src/models/saved/delivery_risk_model.pkl")
    print("Risk model loaded")

    models["anomaly"] = joblib.load("src/models/saved/anomaly_model.pkl")
    print("Anomaly model loaded")

    models["lstm"] = tf.keras.models.load_model("src/models/saved/best_lstm_model.keras")
    models["lstm_scaler"] = joblib.load("src/models/saved/lstm_scaler.pkl")
    with open("src/models/saved/lstm_features.json") as f:
        models["lstm_features"] = json.load(f)
    print("LSTM model loaded")

    models["supplier_risk"] = pd.read_csv("data/processed/supplier_risk_scores.csv")
    print("Supplier risk scores loaded")

    # FinBERT — optional, loads only if transformers supports TF
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        models["sentiment_tokenizer"] = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        models["sentiment_model"] = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        models["sentiment_available"] = True
        print("FinBERT loaded")
    except Exception as e:
        models["sentiment_available"] = False
        models["sentiment_tokenizer"] = None
        models["sentiment_model"] = None
        print(f"FinBERT not available in this environment: {e}")

    print("\nAll models ready!")
    yield
    models.clear()


app = FastAPI(
    title="SmartChain API",
    description="AI-powered supply chain intelligence platform",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():
    return {
        "status": "SmartChain API is running",
        "models": {
            "delivery_risk"   : "active",
            "anomaly"         : "active",
            "demand_forecast" : "active",
            "supplier_risk"   : "active",
            "sentiment"       : "active" if models.get("sentiment_available") else "unavailable"
        }
    }


@app.post("/predict-risk")
def predict_risk(data: dict):
    try:
        df = pd.DataFrame([data])

       
        features = [
            "quantity", "sales", "profit",
            "scheduled_shipping_days",
            "shipping_mode", "market",
            "order_region", "category"
        ]

        X = df[features]
        prediction = models["risk"].predict(X)[0]
        probability = models["risk"].predict_proba(X)[0].max()

        return {
            "risk_level" : int(prediction),
            "risk_label" : "High Risk" if prediction == 1 else "Low Risk",
            "confidence" : round(float(probability), 3)
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/detect-anomaly")
def detect_anomaly(data: dict):
    try:
        df = pd.DataFrame([data])
        features = [
            "quantity", "sales", "profit",
            "delay_days", "is_late",
            "actual_shipping_days", "scheduled_shipping_days"
        ]
        X = df[features].values
        prediction = models["anomaly"].predict(X)[0]
        score = models["anomaly"].decision_function(X)[0]

        return {
            "is_anomaly"    : bool(prediction == -1),
            "label"         : "Anomaly" if prediction == -1 else "Normal",
            "anomaly_score" : round(float(score), 4)
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/forecast-demand")
def forecast_demand(data: dict):
    try:
        last_30 = data.get("last_30_days", [])
        if len(last_30) != 30:
            return {"error": "Please provide exactly 30 days of sales data"}

        sales        = np.array(last_30)
        month        = data.get("month", 6)
        day_of_week  = data.get("day_of_week", 0)
        day_of_year  = data.get("day_of_year", 180)
        lag_7        = sales[-7]
        lag_14       = sales[-14]
        rolling_mean_7 = sales[-7:].mean()

        features = np.array([[
            sales[-1], month, day_of_week,
            day_of_year, lag_7, lag_14, rolling_mean_7
        ]])

        scaled = models["lstm_scaler"].transform(features)
        X = scaled.reshape(1, 1, scaled.shape[1])
        prediction_scaled = models["lstm"].predict(X, verbose=0)

        dummy = np.zeros((1, len(models["lstm_features"])))
        dummy[0, 0] = prediction_scaled[0, 0]
        predicted_sales = models["lstm_scaler"].inverse_transform(dummy)[0, 0]

        return {
            "predicted_demand" : round(float(predicted_sales), 2),
            "unit"             : "daily sales units"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/supplier-risk")
def get_supplier_risk():
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
    try:
        df = models["supplier_risk"]
        supplier = df[df["Supplier"].str.lower() == supplier_name.lower()]
        if supplier.empty:
            return {"error": f"Supplier '{supplier_name}' not found"}
        return supplier.to_dict(orient="records")[0]
    except Exception as e:
        return {"error": str(e)}


@app.get("/sentiment/{company_name}")
def get_sentiment(company_name: str):

    # Check if FinBERT is available
    if not models.get("sentiment_available"):
        return {
            "error": "Sentiment analysis is running locally only. FinBERT requires PyTorch which is unavailable in this deployment environment."
        }

    try:
        NEWS_API_KEY = os.getenv("NEWS_API_KEY")
        url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        data = response.json()

        if data.get("status") != "ok":
            return {"error": data.get("message")}

        articles = data.get("articles", [])
        if not articles:
            return {"error": f"No news found for {company_name}"}

        import torch
        tokenizer = models["sentiment_tokenizer"]
        finbert   = models["sentiment_model"]
        label_map = {0: "positive", 1: "negative", 2: "neutral"}

        results = []
        for article in articles:
            headline = article.get("title", "")
            if not headline:
                continue

            inputs  = tokenizer(headline, return_tensors="pt", truncation=True, max_length=512)
            outputs = finbert(**inputs)
            probs   = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
            label   = label_map[int(np.argmax(probs))]

            results.append({
                "headline"   : headline,
                "sentiment"  : label,
                "confidence" : round(float(np.max(probs)), 3),
                "published"  : article.get("publishedAt", "")
            })

        labels  = [r["sentiment"] for r in results]
        overall = max(set(labels), key=labels.count)

        return {
            "company"          : company_name,
            "overall_sentiment": overall,
            "total_articles"   : len(results),
            "results"          : results
        }
    except Exception as e:
        return {"error": str(e)}