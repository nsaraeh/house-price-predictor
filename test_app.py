"""Quick tests: python -m pytest -q   (or just: python test_app.py)"""
from app import app

GOOD = {"OverallQual": 7, "GrLivArea": 1800, "TotalBsmtSF": 1000, "GarageCars": 2,
        "FullBath": 2, "YearBuilt": 2003, "LotArea": 9500, "Fireplaces": 1,
        "Neighborhood": "CollgCr", "KitchenQual": "Gd"}
c = app.test_client()


def test_home_page():
    r = c.get("/")
    assert r.status_code == 200 and b"House Price Predictor" in r.data


def test_health():
    assert c.get("/health").get_json() == {"status": "ok"}


def test_api_prediction_is_sensible():
    r = c.post("/api/predict", json=GOOD)
    assert r.status_code == 200
    assert 150_000 < r.get_json()["price"] < 350_000


def test_better_house_costs_more():
    cheap = c.post("/api/predict", json=GOOD | {"OverallQual": 3, "GrLivArea": 900}).get_json()
    rich = c.post("/api/predict", json=GOOD | {"OverallQual": 9, "GrLivArea": 2800}).get_json()
    assert rich["price"] > cheap["price"]


def test_form_post_shows_price():
    r = c.post("/predict", data=GOOD)
    assert r.status_code == 200 and b"Estimated sale price" in r.data and b"$" in r.data


def test_validation_errors():
    assert c.post("/api/predict", json=GOOD | {"OverallQual": 99}).status_code == 400
    assert c.post("/api/predict", json=GOOD | {"Neighborhood": "Nowhere"}).status_code == 400
    assert c.post("/api/predict", json={"GrLivArea": "abc"}).status_code == 400
    assert c.post("/api/predict", data="not json").status_code == 400
    assert c.post("/predict", data=GOOD | {"LotArea": ""}).status_code == 400


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
