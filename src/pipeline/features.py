import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

demand_df = pd.read_csv("data/processed/demand_cleaned.csv", parse_dates=["Date"])
supply_df = pd.read_csv("data/processed/supply_chain_cleaned.csv")

print("Building demand features")

demand_df = demand_df.sort_values("Date").reset_index(drop=True)

demand_df["month"]       = demand_df["Date"].dt.month
demand_df["day_of_week"] = demand_df["Date"].dt.dayofweek

demand_df["lag_7"] = demand_df["Order_Demand"].shift(7)

demand_df["rolling_mean_7"] = demand_df["Order_Demand"].rolling(7).mean()

demand_df = demand_df.dropna().reset_index(drop=True)

demand_df.to_csv("data/processed/demand_featured.csv", index=False)
print(f"Demand features done — {len(demand_df)} rows")


print("\nBuilding supply chain features")

supply_df["delay_days"] = supply_df["actual_shipping_days"] - supply_df["scheduled_shipping_days"]

supply_df["is_late"] = (supply_df["delay_days"] > 0).astype(int)

# 0 = low risk, 1 = high risk
supply_df["risk_level"] = supply_df["late_delivery_risk"]

supply_df.to_csv("data/processed/supply_featured.csv", index=False)
print(f"Supply features done — {len(supply_df)} rows")

print("\nAll features saved!")