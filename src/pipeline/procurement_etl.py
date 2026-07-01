import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/raw/procurement.csv")
print("Raw data loaded —", df.shape)

df["Order_Date"]    = pd.to_datetime(df["Order_Date"])
df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"])

df["delay_days"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days


df["Defective_Units"] = df["Defective_Units"].fillna(0)


# What percentage of the order was defective
df["defect_rate"] = df["Defective_Units"] / df["Quantity"]


# How much was saved through negotiation
df["price_savings"] = df["Unit_Price"] - df["Negotiated_Price"]
df["savings_pct"] = (df["price_savings"] / df["Unit_Price"]) * 100

#  Convert Compliance to 1/0
df["is_compliant"] = (df["Compliance"] == "Yes").astype(int)


df = df.dropna(subset=["delay_days", "defect_rate"])


df.to_csv("data/processed/procurement_clean.csv", index=False)

print(f"Cleaned data saved — {len(df)} rows")
print("\nColumns:", df.columns.tolist())
print("\nSample:")
print(df[["Supplier", "delay_days", "defect_rate", "is_compliant"]].head(10))