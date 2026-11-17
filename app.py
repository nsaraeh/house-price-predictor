"""
House Price Predictor - Flask web app.

Run locally :  python app.py            (http://127.0.0.1:5000)
Production  :  gunicorn app:app
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).parent
model = joblib.load(ROOT / "model" / "house_model.joblib")
meta = json.loads((ROOT / "model" / "model_meta.json").read_text())

FEATURES = meta["features"]
NEIGHBORHOODS = meta["neighborhoods"]
TYPICAL_ERROR = int(round(meta["test_metrics"]["Test MAE ($)"], -2))

# Allowed values for every input: (label, min, max, default, step, help text)
NUMERIC_FIELDS = {
    "OverallQual": ("Overall quality (1 = poor, 10 = excellent)", 1, 10, 5, 1),
    "GrLivArea": ("Above-ground living area (sq ft)", 300, 6000, 1500, 50),
    "TotalBsmtSF": ("Total basement area (sq ft, 0 = none)", 0, 4000, 800, 50),
    "GarageCars": ("Garage capacity (cars)", 0, 4, 2, 1),
    "FullBath": ("Full bathrooms", 0, 4, 2, 1),
    "YearBuilt": ("Year built", 1870, 2026, 1990, 1),
    "LotArea": ("Lot area (sq ft)", 1000, 60000, 9000, 500),
    "Fireplaces": ("Fireplaces", 0, 3, 1, 1),
}
KITCHEN = {"Ex": "Excellent", "Gd": "Good", "TA": "Typical / Average", "Fa": "Fair"}
NEIGHBORHOOD_NAMES = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale",
    "BrkSide": "Brookside", "ClearCr": "Clear Creek", "CollgCr": "College Creek",
    "Crawfor": "Crawford", "Edwards": "Edwards", "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT and Rail Road", "MeadowV": "Meadow Village",
    "Mitchel": "Mitchell", "NAmes": "North Ames", "NPkVill": "Northpark Villa",
    "NWAmes": "Northwest Ames", "NoRidge": "Northridge", "NridgHt": "Northridge Heights",
    "OldTown": "Old Town", "SWISU": "South & West of Iowa State Univ.",
    "Sawyer": "Sawyer", "SawyerW": "Sawyer West", "Somerst": "Somerset",
    "StoneBr": "Stone Brook", "Timber": "Timberland", "Veenker": "Veenker",
}

app = Flask(__name__)


def parse_and_validate(values: dict):
    """Return (clean_row, errors). `values` can come from a form or from JSON."""
    row, errors = {}, {}
    for name, (label, lo, hi, _default, _step) in NUMERIC_FIELDS.items():
        raw = values.get(name)
        if raw is None or str(raw).strip() == "":
            errors[name] = "This field is required."
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            errors[name] = "Please enter a number."
            continue
        if not lo <= num <= hi:
            errors[name] = f"Must be between {lo:,} and {hi:,}."
            continue
        row[name] = num
    hood = values.get("Neighborhood")
    if hood not in NEIGHBORHOODS:
        errors["Neighborhood"] = "Please choose a neighborhood from the list."
    else:
        row["Neighborhood"] = hood
    kitchen = values.get("KitchenQual")
    if kitchen not in KITCHEN:
        errors["KitchenQual"] = "Please choose a kitchen quality from the list."
    else:
        row["KitchenQual"] = kitchen
    return row, errors


def predict_price(row: dict) -> dict:
    price = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
    return {
        "price": round(price),
        "low": max(0, int(round(price - TYPICAL_ERROR, -2))),
        "high": int(round(price + TYPICAL_ERROR, -2)),
    }


def render_form(values, errors=None, result=None, status=200):
    return render_template(
        "index.html", fields=NUMERIC_FIELDS, kitchen=KITCHEN,
        neighborhoods=[(n, NEIGHBORHOOD_NAMES.get(n, n)) for n in NEIGHBORHOODS],
        values=values, errors=errors or {}, result=result, meta=meta,
        typical_error=TYPICAL_ERROR), status


@app.get("/")
def index():
    defaults = {k: v[3] for k, v in NUMERIC_FIELDS.items()}
    defaults.update(Neighborhood="NAmes", KitchenQual="TA")
    return render_form(defaults)


@app.post("/predict")
def predict_form():
    row, errors = parse_and_validate(request.form)
    if errors:
        return render_form(request.form, errors=errors, status=400)
    return render_form(request.form, result=predict_price(row))


@app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object with the house features."), 400
    row, errors = parse_and_validate(payload)
    if errors:
        return jsonify(error="Invalid input", details=errors), 400
    return jsonify(predict_price(row) | {"currency": "USD", "inputs": row})


@app.get("/api/model-info")
def model_info():
    return jsonify(model=meta["model_name"], features=FEATURES,
                   trained_on_rows=meta["n_training_rows"],
                   test_metrics=meta["test_metrics"])


@app.get("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
