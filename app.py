"""
Flask Web Application for House Price Prediction System.
Provides an interactive valuation calculator, presets, performance metrics,
model diagnostics charts, and batch prediction download.
"""

import json
import os
from flask import Flask, jsonify, render_template, request, send_file
from src.predict import HousePricePredictor

app = Flask(__name__)

# Initialize predictor
MODEL_PATH = "models/house_price_pipeline.joblib"
METRICS_PATH = "models/model_metrics.json"

predictor = HousePricePredictor(MODEL_PATH)

# Load metrics metadata
model_metadata = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        model_metadata = json.load(f)

# Preset property profiles for instant user exploration
PRESETS = {
    "luxury": {
        "name": "Luxury Executive Estate",
        "description": "Premium newly-built mansion with high-end finishes, 3-car garage, and expansive living space.",
        "data": {
            "OverallQual": 9,
            "OverallCond": 6,
            "GrLivArea": 3200,
            "TotalBsmtSF": 1800,
            "1stFlrSF": 1800,
            "2ndFlrSF": 1400,
            "GarageCars": 3,
            "GarageArea": 850,
            "YearBuilt": 2018,
            "YearRemodAdd": 2019,
            "FullBath": 3,
            "HalfBath": 1,
            "BedroomAbvGr": 4,
            "TotRmsAbvGrd": 10,
            "Fireplaces": 2,
            "LotArea": 16500,
            "Neighborhood": "NridgHt",
            "KitchenQual": "Ex",
            "ExterQual": "Ex",
            "BsmtQual": "Ex",
            "BldgType": "1Fam",
            "HouseStyle": "2Story",
        },
    },
    "suburban": {
        "name": "Suburban Family Home",
        "description": "Popular Ames suburban 2-story home with 3 bedrooms, finished basement, and 2-car garage.",
        "data": {
            "OverallQual": 7,
            "OverallCond": 5,
            "GrLivArea": 1850,
            "TotalBsmtSF": 950,
            "1stFlrSF": 950,
            "2ndFlrSF": 900,
            "GarageCars": 2,
            "GarageArea": 550,
            "YearBuilt": 2004,
            "YearRemodAdd": 2005,
            "FullBath": 2,
            "HalfBath": 1,
            "BedroomAbvGr": 3,
            "TotRmsAbvGrd": 7,
            "Fireplaces": 1,
            "LotArea": 9500,
            "Neighborhood": "CollgCr",
            "KitchenQual": "Gd",
            "ExterQual": "Gd",
            "BsmtQual": "Gd",
            "BldgType": "1Fam",
            "HouseStyle": "2Story",
        },
    },
    "starter": {
        "name": "Cozy Starter Cottage",
        "description": "Affordable 1-story classic home with solid construction, modest footprint, and single garage.",
        "data": {
            "OverallQual": 5,
            "OverallCond": 6,
            "GrLivArea": 1050,
            "TotalBsmtSF": 850,
            "1stFlrSF": 1050,
            "2ndFlrSF": 0,
            "GarageCars": 1,
            "GarageArea": 320,
            "YearBuilt": 1968,
            "YearRemodAdd": 1995,
            "FullBath": 1,
            "HalfBath": 0,
            "BedroomAbvGr": 2,
            "TotRmsAbvGrd": 5,
            "Fireplaces": 0,
            "LotArea": 7200,
            "Neighborhood": "NAmes",
            "KitchenQual": "TA",
            "ExterQual": "TA",
            "BsmtQual": "TA",
            "BldgType": "1Fam",
            "HouseStyle": "1Story",
        },
    },
    "fixer": {
        "name": "Historic Fixer-Upper",
        "description": "Turn-of-the-century older home with character requiring renovation in a traditional neighborhood.",
        "data": {
            "OverallQual": 4,
            "OverallCond": 4,
            "GrLivArea": 1250,
            "TotalBsmtSF": 650,
            "1stFlrSF": 750,
            "2ndFlrSF": 500,
            "GarageCars": 1,
            "GarageArea": 240,
            "YearBuilt": 1925,
            "YearRemodAdd": 1950,
            "FullBath": 1,
            "HalfBath": 0,
            "BedroomAbvGr": 3,
            "TotRmsAbvGrd": 6,
            "Fireplaces": 0,
            "LotArea": 6000,
            "Neighborhood": "OldTown",
            "KitchenQual": "Fa",
            "ExterQual": "Fa",
            "BsmtQual": "Fa",
            "BldgType": "1Fam",
            "HouseStyle": "1.5Fin",
        },
    },
}


@app.route("/")
def index():
    return render_template(
        "index.html",
        metadata=model_metadata,
        presets=PRESETS,
    )


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json(force=True)
        # Parse numeric types appropriately
        cleaned_data = {}
        for key, val in data.items():
            if val is None or val == "":
                continue
            try:
                if "." in str(val):
                    cleaned_data[key] = float(val)
                else:
                    cleaned_data[key] = int(val)
            except ValueError:
                cleaned_data[key] = str(val)

        # Synchronize GrLivArea if 1stFlrSF + 2ndFlrSF provided
        first_flr = cleaned_data.get("1stFlrSF", 0)
        sec_flr = cleaned_data.get("2ndFlrSF", 0)
        if "GrLivArea" not in cleaned_data or cleaned_data["GrLivArea"] <= 0:
            cleaned_data["GrLivArea"] = first_flr + sec_flr

        result = predictor.predict_single(cleaned_data)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/presets")
def get_presets():
    return jsonify(PRESETS)


@app.route("/api/metrics")
def get_metrics():
    return jsonify(model_metadata)


@app.route("/download-submission")
def download_submission():
    sub_path = "submission_predicted.csv"
    if os.path.exists(sub_path):
        return send_file(sub_path, as_attachment=True, download_name="submission_predicted.csv")
    return "Submission file not found. Run pipeline first.", 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
