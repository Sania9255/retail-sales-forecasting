import pandas as pd

# Load datasets
train = pd.read_csv("data/walmart/train.csv")
features = pd.read_csv("data/walmart/features.csv")
stores = pd.read_csv("data/walmart/stores.csv")

# Merge datasets
df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
df = df.merge(stores, on="Store", how="left")

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# Encode categories
df["Type"] = df["Type"].map({"A": 1, "B": 2, "C": 3})

# Fill missing values if any
df = df.fillna(method="ffill")

# Save processed data
df.to_csv("data/processed/processed.csv", index=False)

print("Processing Completed ✔ Data saved in data/processed/processed.csv")