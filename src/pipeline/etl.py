import pandas as pd
import os

RAW_PATH       = "data/raw/"
PROCESSED_PATH = "data/processed/"

def clean_demand_data():
    print("Cleaning demand data")

    df = pd.read_csv(RAW_PATH + "demand.csv")

    df = df.dropna(subset=["Date"])

    df["Date"] = pd.to_datetime(df["Date"])

    df["Order_Demand"] = pd.to_numeric(df["Order_Demand"], errors="coerce")
    df = df.dropna(subset=["Order_Demand"])
    df = df[df["Order_Demand"] > 0]

    df = df.sort_values("Date").reset_index(drop=True)

    df.to_csv(PROCESSED_PATH + "demand_cleaned.csv", index=False)
    print(f"Demand data cleaned — {len(df)} rows saved.")

    return df


def clean_supply_chain_data():
    print("Cleaning supply chain data...")

    df = pd.read_csv(
        RAW_PATH + "supply_chain.csv",
        encoding="unicode_escape"
    )

    # Keep only the columns we need
    columns_we_need = [
        "Late_delivery_risk",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Delivery Status",
        "Order Item Quantity",
        "Sales",
        "Order Profit Per Order",
        "Shipping Mode",
        "Market",
        "Order Region",
        "Product Name",
        "Category Name"
    ]
    df = df[columns_we_need]

    df = df.rename(columns={
        "Days for shipping (real)"       : "actual_shipping_days",
        "Days for shipment (scheduled)"  : "scheduled_shipping_days",
        "Delivery Status"                : "delivery_status",
        "Late_delivery_risk"             : "late_delivery_risk",
        "Order Item Quantity"            : "quantity",
        "Sales"                          : "sales",
        "Order Profit Per Order"         : "profit",
        "Shipping Mode"                  : "shipping_mode",
        "Market"                         : "market",
        "Order Region"                   : "order_region",
        "Product Name"                   : "product_name",
        "Category Name"                  : "category"
    })

    df = df.dropna()
    #New column
    df["shipping_delay"] = df["actual_shipping_days"] - df["scheduled_shipping_days"]

    # Save cleaned file
    df.to_csv(PROCESSED_PATH + "supply_chain_cleaned.csv", index=False)
    print(f"Supply chain data cleaned — {len(df)} rows saved.")

    return df


if __name__ == "__main__":
    clean_demand_data()
    clean_supply_chain_data()
    print("\nAll data cleaned and saved to data/processed/")