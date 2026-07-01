import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

demand_df = pd.read_csv("data/raw/demand.csv")
supply_df = pd.read_csv("data/raw/supply_chain.csv", encoding="unicode_escape")

print("Shape:", demand_df.shape)
print("\nColumns:", demand_df.columns.tolist())
print("\nFirst 3 rows:")
print(demand_df.head(3))
print("\nMissing values:")
print(demand_df.isnull().sum())

print("\n====== SUPPLY CHAIN DATASET ======")
print("Shape:", supply_df.shape)
print("\nColumns:", supply_df.columns.tolist())
print("\nFirst 3 rows:")
print(supply_df.head(3))
print("\nMissing values:")
print(supply_df.isnull().sum())