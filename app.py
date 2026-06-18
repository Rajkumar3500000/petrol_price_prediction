from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import pickle

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "petrol.csv")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

df = pd.read_csv(CSV_PATH)
df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
df = df.sort_values("Date").reset_index(drop=True)

MIN_YEAR = int(df["Date"].dt.year.min())

@app.route("/")
def home():
    return render_template("index.html", min_year=MIN_YEAR)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        year = int(request.form["year"])
        month = int(request.form["month"])

        if year < MIN_YEAR:
            return render_template(
                "index.html",
                min_year=MIN_YEAR,
                prediction_text=f"Year must be {MIN_YEAR} or later (no data before that)."
            )

        if month < 1 or month > 12:
            return render_template(
                "index.html",
                min_year=MIN_YEAR,
                prediction_text="⚠️ Month must be between 1 and 12."
            )

        prediction = model.predict([[year, month]])

        return render_template(
            "index.html",
            min_year=MIN_YEAR,
            prediction_text=f"Predicted Petrol Price: ₹ {round(prediction[0], 2)}"
        )

    except ValueError:
        return render_template(
            "index.html",
            min_year=MIN_YEAR,
            prediction_text="Please enter valid numbers for year and month."
        )

@app.route("/chart-data")
def chart_data():
    labels = df["Date"].dt.strftime("%b %Y").tolist()
    values = df["Avg Petrol Rate India (₹/L)"].tolist()
    return jsonify({"labels": labels, "values": values})

if __name__ == "__main__":
    app.run(debug=True)
