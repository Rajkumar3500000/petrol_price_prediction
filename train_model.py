import pandas as pd 
from sklearn.linear_model import LinearRegression
import pickle
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "petrol.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

df = pd.read_csv(CSV_PATH)

df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")

df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month

X = df[["year","month"]]
y = df["Avg Petrol Rate India (₹/L)"]

model = LinearRegression()
model.fit(X,y)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("Model Saved Successfully")
