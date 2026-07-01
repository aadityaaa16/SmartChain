# 🚀 SmartChain — AI-Powered Supply Chain Intelligence Platform

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FinBERT-yellow?style=for-the-badge&logo=huggingface)
![Railway](https://img.shields.io/badge/Deploy-Railway-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

## 📌 Overview

**SmartChain** is an end-to-end AI-powered Supply Chain Intelligence Platform that helps organizations make smarter operational decisions through Machine Learning, Deep Learning, Explainable AI, and Financial NLP.

Instead of relying on historical dashboards, SmartChain predicts future demand, identifies risky suppliers, detects unusual transactions, optimizes delivery routes, and monitors supplier news sentiment in one unified platform.

The project combines modern AI techniques with an interactive web application using **FastAPI** and **Streamlit**.

---

# ❓ Problem Statement

Supply chains generate massive volumes of operational data every day, but organizations often struggle to answer questions like:

- Which orders are likely to become risky?
- Which orders are abnormal and require investigation?
- How much inventory will be needed next month?
- Are supplier-related news events affecting procurement risk?


Traditional reporting systems provide historical insights but fail to provide predictive intelligence.

SmartChain addresses these challenges by integrating multiple AI models into a single intelligent platform.

---

# 💡 Solution

SmartChain leverages multiple Machine Learning and AI models to solve real-world supply chain problems.

The platform provides:

✅ Delivery Risk Prediction

✅ Supplier Risk Scoring

✅ Demand Forecasting

✅ Anomaly Detection

✅ Financial News Sentiment Analysis


through an interactive dashboard backed by production-ready REST APIs.

---

# 🎯 Key Features

### 📦 Delivery Risk Prediction

Predicts whether an order has a high probability of delayed delivery using a Random Forest classifier optimized with Optuna.

**Technologies**

- Random Forest
- Optuna Hyperparameter Optimization
- SHAP Explainability

---

### 📈 Demand Forecasting

Forecasts future product demand using Deep Learning.

**Technologies**

- TensorFlow
- Keras
- LSTM
- Time Series Forecasting

---

### ⚠️ Anomaly Detection

Detects suspicious orders and unusual business transactions.

**Technologies**

- Isolation Forest
- SHAP Explainability

---

### 🏭 Supplier Risk Dashboard

Ranks suppliers using procurement KPIs such as:

- Delay Rate
- Defect Rate
- Compliance Score

Provides an overall supplier risk score and ranking.

---

### 📰 Financial Sentiment Analysis

Analyzes recent supplier-related financial news using FinBERT.

Detects whether news is:

- Positive
- Neutral
- Negative

to estimate supplier reputation and business impact.


---

# 🧠 AI Models

| Module | Algorithm | Purpose |
|----------|-----------|----------|
| Delivery Risk | Random Forest + Optuna | Predict delayed deliveries |
| Demand Forecasting | LSTM (TensorFlow) | Forecast future demand |
| Anomaly Detection | Isolation Forest | Detect abnormal orders |
| Explainability | SHAP | Explain model predictions |
| Supplier Risk | KPI Weighted Scoring | Rank supplier reliability |
| News Sentiment | FinBERT | Financial NLP |

---

# 📊 Model Performance

| Model | Performance |
|--------|------------|
| Random Forest Risk Prediction | **70% Accuracy** |
| LSTM Demand Forecasting | **MAPE ≈ 21.8%** |
| Isolation Forest | **9,026 anomalies detected** |
| Supplier Ranking | **5 suppliers evaluated** |
| FinBERT | Live Financial Sentiment Analysis |

---

# 🏗 System Architecture

```
                Supply Chain Data
                        │
                        ▼
               ETL & Data Cleaning
                        │
                        ▼
             Feature Engineering Pipeline
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
 Random Forest      Isolation       LSTM Forecast
   Risk Model        Forest            Model
        │               │                │
        └───────────────┼────────────────┘
                        │
                        ▼
                 FastAPI Backend
                        │
               REST API Endpoints
                        │
                        ▼
              Streamlit Dashboard
                        │
                        ▼
                     End User
```

---

# 🛠 Tech Stack

## Programming

- Python

---

## Machine Learning

- Scikit-learn
- XGBoost
- Optuna
- SHAP

---

## Deep Learning

- TensorFlow
- Keras

---

## NLP

- HuggingFace Transformers
- FinBERT

---

## Backend

- FastAPI
- Uvicorn
- Pydantic

---

## Frontend

- Streamlit
- Plotly
- Matplotlib

---

## Database

- MySQL
- SQLAlchemy


---

## Deployment

- Railway
- Streamlit Community Cloud

---

## Version Control

- Git
- GitHub

---

# 📂 Project Structure

```
smartchain
│
├── app/
│   └── main.py                 # Streamlit Frontend
│
├── src/
│   ├── api/                    # FastAPI APIs
│   ├── models/                 # Trained ML Models
│   ├── pipeline/               # ETL & Feature Engineering
│   └── utils/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

# 📡 REST API

| Endpoint | Description |
|------------|-------------|
| `/predict-risk` | Delivery Risk Prediction |
| `/detect-anomaly` | Detect Abnormal Orders |
| `/forecast-demand` | Predict Future Demand |
| `/supplier-risk` | Supplier Ranking |
| `/sentiment/{company}` | Financial News Sentiment |

---

# 🌐 Live Demo

## Streamlit Dashboard

> https://smartchain-3yvhilyezabaqrrtvqb5vh.streamlit.app/

---

## FastAPI Documentation

> https://web-production-e55a2.up.railway.app/

---

# 📊 Datasets

- Product Demand Forecasting Dataset
- DataCo Smart Supply Chain Dataset
- Procurement KPI Dataset
- Store Item Demand Forecasting Dataset

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/aadityaaa16/SmartChain.git
```

Move into the project

```bash
cd SmartChain
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Start FastAPI

```bash
uvicorn src.api.main:app --reload
```

Open another terminal and start Streamlit

```bash
streamlit run app/main.py
```



---

# 👨‍💻 Author

**Aditya Singh**

GitHub: https://github.com/aadityaaa16

---

# ⭐ If you found this project useful...

Please consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future development.

---