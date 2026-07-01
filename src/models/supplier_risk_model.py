import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/processed/procurement_clean.csv")
print("Data loaded —", df.shape)


suppliers = df.groupby("Supplier").agg(
    total_orders    = ("PO_ID", "count"),
    avg_delay_days  = ("delay_days", "mean"),
    avg_defect_rate = ("defect_rate", "mean"),
    compliance_rate = ("is_compliant", "mean")
).reset_index()

print("\nSupplier performance:")
print(suppliers)


# 0 = best, 1 = worst
suppliers["delay_score"]  = suppliers["avg_delay_days"] / suppliers["avg_delay_days"].max()
suppliers["defect_score"] = suppliers["avg_defect_rate"] / suppliers["avg_defect_rate"].max()
suppliers["compliance_score"] = 1 - suppliers["compliance_rate"]

# Combine into one risk score out of 100 

suppliers["risk_score"] = (
    suppliers["delay_score"] * 40 +
    suppliers["defect_score"] * 40 +
    suppliers["compliance_score"] * 20
).round(1)

def label_risk(score):
    if score < 30:
        return "Low Risk"
    elif score < 60:
        return "Medium Risk"
    else:
        return "High Risk"

suppliers["risk_category"] = suppliers["risk_score"].apply(label_risk)

#  Sort riskiest supplier first 
suppliers = suppliers.sort_values("risk_score", ascending=False)

print("\nFinal Supplier Risk Ranking:")
print(suppliers[["Supplier", "risk_score", "risk_category"]])


suppliers.to_csv("data/processed/supplier_risk_scores.csv", index=False)
print("\nSaved to data/processed/supplier_risk_scores.csv")