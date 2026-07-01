import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

#  API base URL
API_URL = "http://127.0.0.1:8000"

# --- Page config ---
st.set_page_config(
    page_title="SmartChain",
    page_icon="🔗",
    layout="wide"
)

# --- Sidebar navigation ---
st.sidebar.title("🔗 SmartChain")
st.sidebar.markdown("AI-Powered Supply Chain Intelligence")

page = st.sidebar.radio("Navigate", [
    "Overview",
    "Delivery Risk",
    "Anomaly Detection",
    "Demand Forecast",
    "Supplier Risk",
    "Sentiment Analysis"
])

st.sidebar.markdown("---")
st.sidebar.caption("Built with FastAPI + Streamlit")


# ========================
# PAGE 1 — OVERVIEW
# ========================
if page == "Overview":
    st.title("🔗 SmartChain Dashboard")
    st.markdown("AI-powered supply chain intelligence platform")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("ML Models", "4", "Active")
    with col2:
        st.metric("Suppliers Monitored", "5", "Live")
    with col3:
        st.metric("Orders Analyzed", "180,519", "Total")
    with col4:
        st.metric("Anomalies Detected", "9,026", "Flagged")

    st.markdown("---")

    # Show supplier risk summary on overview
    st.subheader("Supplier Risk Summary")
    try:
        response = requests.get(f"{API_URL}/supplier-risk")
        data = response.json()
        df = pd.DataFrame(data["suppliers"])

        fig = px.bar(
            df,
            x="Supplier",
            y="risk_score",
            color="risk_category",
            color_discrete_map={
                "Low Risk"    : "#1D9E75",
                "Medium Risk" : "#EF9F27",
                "High Risk"   : "#E24B4A"
            },
            title="Supplier Risk Scores"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load supplier data: {e}")

    st.markdown("---")
    st.subheader("About SmartChain")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **What SmartChain does:**
        - Predicts delivery risk for incoming orders
        - Detects anomalies in order patterns
        - Forecasts future product demand
        - Scores supplier reliability
        - Analyzes supplier news sentiment
        """)
    with col2:
        st.markdown("""
        **Tech stack:**
        - Random Forest + Optuna + SHAP
        - Isolation Forest
        - LSTM (TensorFlow/Keras)
        - FinBERT (HuggingFace)
        - FastAPI + Streamlit
        """)


# ========================
# PAGE 2 — DELIVERY RISK
# ========================
elif page == "Delivery Risk":
    st.title("📦 Delivery Risk Predictor")
    st.markdown("Predict whether an order is at risk of late delivery")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        quantity  = st.number_input("Order Quantity", min_value=1, value=100)
        sales     = st.number_input("Sales Value ($)", min_value=0.0, value=500.0)
        profit    = st.number_input("Profit ($)", min_value=-1000.0, value=50.0)
        scheduled = st.number_input("Scheduled Shipping Days", min_value=1, value=4)

    with col2:
        shipping_mode = st.selectbox("Shipping Mode", [
            "Standard Class", "Second Class",
            "First Class", "Same Day"
        ])
        market       = st.selectbox("Market", [
            "Europe", "LATAM", "North America",
            "Pacific Asia", "Africa"
        ])
        order_region = st.selectbox("Order Region", [
            "Western Europe", "Central America",
            "South Asia", "Southeast Asia",
            "North America", "East Africa"
        ])
        category     = st.selectbox("Category", [
            "Office Supplies", "Furniture",
            "Technology", "Clothing", "Sports"
        ])

    if st.button("Predict Risk", type="primary"):
        payload = {
            "quantity"                : quantity,
            "sales"                   : sales,
            "profit"                  : profit,
            "scheduled_shipping_days" : scheduled,
            "shipping_mode"           : shipping_mode,
            "market"                  : market,
            "order_region"            : order_region,
            "category"                : category
        }

        try:
            response = requests.post(f"{API_URL}/predict-risk", json=payload)
            result   = response.json()

            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown("---")
                col1, col2, col3 = st.columns(3)

                with col1:
                    label = result["risk_label"]
                    if label == "High Risk":
                        st.error(f"🔴 {label}")
                    else:
                        st.success(f"🟢 {label}")

                with col2:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']*100:.1f}%"
                    )

                with col3:
                    st.metric("Risk Level", result["risk_level"])

        except Exception as e:
            st.error(f"API error: {e}")


# ========================
# PAGE 3 — ANOMALY DETECTION
# ========================
elif page == "Anomaly Detection":
    st.title("🚨 Anomaly Detection")
    st.markdown("Detect unusual patterns in supply chain orders")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        quantity       = st.number_input("Order Quantity",       min_value=1,      value=100)
        sales          = st.number_input("Sales Value ($)",      min_value=0.0,    value=500.0)
        profit         = st.number_input("Profit ($)",           min_value=-5000.0,value=50.0)

    with col2:
        actual_days    = st.number_input("Actual Shipping Days",     min_value=0, value=5)
        scheduled_days = st.number_input("Scheduled Shipping Days",  min_value=1, value=4)
        delay_days     = actual_days - scheduled_days
        is_late        = 1 if delay_days > 0 else 0

        st.info(f"Delay days: {delay_days} | Is late: {'Yes' if is_late else 'No'}")

    if st.button("Detect Anomaly", type="primary"):
        payload = {
            "quantity"                : quantity,
            "sales"                   : sales,
            "profit"                  : profit,
            "delay_days"              : delay_days,
            "is_late"                 : is_late,
            "actual_shipping_days"    : actual_days,
            "scheduled_shipping_days" : scheduled_days
        }

        try:
            response = requests.post(f"{API_URL}/detect-anomaly", json=payload)
            result   = response.json()

            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    if result["is_anomaly"]:
                        st.error("🚨 ANOMALY DETECTED")
                        st.markdown("This order shows unusual patterns")
                    else:
                        st.success("✅ NORMAL ORDER")
                        st.markdown("This order looks normal")

                with col2:
                    st.metric(
                        "Anomaly Score",
                        round(result["anomaly_score"], 4)
                    )
                    st.caption("More negative = more unusual")

        except Exception as e:
            st.error(f"API error: {e}")


# ========================
# PAGE 4 — DEMAND FORECAST
# ========================
elif page == "Demand Forecast":
    st.title("📈 Demand Forecasting")
    st.markdown("LSTM model predicts next day demand based on last 30 days")
    st.markdown("---")

    st.subheader("Enter last 30 days of sales")
    st.caption("Enter one value per day — or use the sample data below")

    # Sample data button
    if st.button("Load Sample Data"):
        sample = [45,48,52,49,51,53,47,50,55,52,
                  48,51,49,53,56,50,48,52,54,49,
                  51,53,47,50,55,52,48,51,49,53]
        st.session_state["sample_sales"] = sample

    # Input fields for 30 days
    sales_input = []
    cols = st.columns(6)

    for i in range(30):
        default = 50
        if "sample_sales" in st.session_state:
            default = st.session_state["sample_sales"][i]

        val = cols[i % 6].number_input(
            f"Day {i+1}",
            min_value=0,
            value=default,
            key=f"day_{i}"
        )
        sales_input.append(val)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        month       = st.slider("Month", 1, 12, 6)
    with col2:
        day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 0)
    with col3:
        day_of_year = st.slider("Day of Year", 1, 365, 180)

    if st.button("Forecast Tomorrow", type="primary"):
        payload = {
            "last_30_days": sales_input,
            "month"       : month,
            "day_of_week" : day_of_week,
            "day_of_year" : day_of_year
        }

        try:
            response = requests.post(
                f"{API_URL}/forecast-demand",
                json=payload
            )
            result = response.json()

            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown("---")
                st.metric(
                    "Predicted Tomorrow's Demand",
                    f"{result['predicted_demand']:.0f} units"
                )

                # Plot last 30 days + prediction
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, 31)),
                    y=sales_input,
                    name="Last 30 days",
                    line=dict(color="#1D9E75")
                ))
                fig.add_trace(go.Scatter(
                    x=[31],
                    y=[result["predicted_demand"]],
                    name="Predicted",
                    mode="markers",
                    marker=dict(size=12, color="#E24B4A")
                ))
                fig.update_layout(
                    title="Sales History + Forecast",
                    xaxis_title="Day",
                    yaxis_title="Sales Units"
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"API error: {e}")


# ========================
# PAGE 5 — SUPPLIER RISK
# ========================
elif page == "Supplier Risk":
    st.title("🏭 Supplier Risk Dashboard")
    st.markdown("Pre-computed risk scores based on procurement KPIs")
    st.markdown("---")

    try:
        response = requests.get(f"{API_URL}/supplier-risk")
        data     = response.json()
        df       = pd.DataFrame(data["suppliers"])

        # Color-coded risk table
        def color_risk(val):
            if val == "High Risk":
                return "background-color: #FCEBEB; color: #791F1F"
            elif val == "Medium Risk":
                return "background-color: #FAEEDA; color: #633806"
            else:
                return "background-color: #EAF3DE; color: #27500A"

        st.subheader("All Suppliers")
        styled = df.style.map(
            color_risk,
            subset=["risk_category"]
        )
        st.dataframe(styled, use_container_width=True)

        st.markdown("---")

        # Risk score bar chart
        fig = px.bar(
            df.sort_values("risk_score", ascending=True),
            x="risk_score",
            y="Supplier",
            orientation="h",
            color="risk_category",
            color_discrete_map={
                "Low Risk"    : "#1D9E75",
                "Medium Risk" : "#EF9F27",
                "High Risk"   : "#E24B4A"
            },
            title="Supplier Risk Ranking"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Single supplier detail
        st.subheader("Supplier Detail")
        selected = st.selectbox(
            "Select a supplier",
            df["Supplier"].tolist()
        )

        supplier_data = df[df["Supplier"] == selected].iloc[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Score", f"{supplier_data['risk_score']}/100")
        with col2:
            st.metric("Avg Delay Days", f"{supplier_data['avg_delay_days']:.1f}")
        with col3:
            st.metric("Compliance Rate", f"{supplier_data['compliance_rate']*100:.1f}%")

    except Exception as e:
        st.error(f"Could not load supplier data: {e}")


# ========================
# PAGE 6 — SENTIMENT ANALYSIS
# ========================
elif page == "Sentiment Analysis":
    st.title("📰 Supplier Sentiment Analysis")
    st.markdown("Real-time news sentiment using FinBERT")
    st.markdown("---")

    company = st.text_input(
        "Enter company name",
        placeholder="e.g. Tata Steel, Maersk, Samsung"
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not company:
            st.warning("Please enter a company name")
        else:
            with st.spinner(f"Fetching news and analyzing sentiment for {company}..."):
                try:
                    response = requests.get(
                        f"{API_URL}/sentiment/{company}",
                        timeout=30
                    )
                    result = response.json()

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.markdown("---")

                        col1, col2 = st.columns(2)

                        with col1:
                            overall = result["overall_sentiment"]
                            if overall == "positive":
                                st.success(f"Overall Sentiment: 🟢 POSITIVE")
                            elif overall == "negative":
                                st.error(f"Overall Sentiment: 🔴 NEGATIVE")
                            else:
                                st.info(f"Overall Sentiment: 🟡 NEUTRAL")

                        with col2:
                            st.metric(
                                "Articles Analyzed",
                                result["total_articles"]
                            )

                        st.markdown("---")
                        st.subheader("Headline Analysis")

                        for item in result["results"]:
                            col1, col2, col3 = st.columns([4, 1, 1])
                            with col1:
                                st.write(item["headline"])
                            with col2:
                                if item["sentiment"] == "positive":
                                    st.success(item["sentiment"])
                                elif item["sentiment"] == "negative":
                                    st.error(item["sentiment"])
                                else:
                                    st.info(item["sentiment"])
                            with col3:
                                st.write(f"{item['confidence']*100:.0f}%")

                        # Sentiment distribution chart
                        labels = [r["sentiment"] for r in result["results"]]
                        label_counts = pd.Series(labels).value_counts().reset_index()
                        label_counts.columns = ["sentiment", "count"]

                        fig = px.pie(
                            label_counts,
                            names="sentiment",
                            values="count",
                            color="sentiment",
                            color_discrete_map={
                                "positive" : "#1D9E75",
                                "negative" : "#E24B4A",
                                "neutral"  : "#EF9F27"
                            },
                            title=f"Sentiment Distribution — {company}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"API error: {e}")