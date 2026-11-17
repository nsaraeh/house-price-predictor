# House Price Predictor

Bonus challenge: (1) compare Ridge / Lasso with Linear Regression, (2) deploy the model as a Flask web app.

## Project layout
| Path | What it is |
|---|---|
| `train.py` | Trains + compares all models, saves the app model |
| `app.py`, `templates/index.html` | Flask web app (form + JSON API) |
| `model/house_model.joblib` | Saved model (Lasso pipeline, predicts dollars) |
| `reports/` | Comparison tables and charts |
| `test_app.py` | Automated tests for the app |
| `Dockerfile`, `Procfile`, `requirements.txt` | Deployment files |
| `data/train.csv` | Kaggle House Prices training data |

## Run locally
```bash
pip install -r requirements.txt
python train.py        # optional: re-creates the model + reports (~1 min)
python app.py          # open http://127.0.0.1:5000
python test_app.py     # run the tests
```

## Deploy
**Render / Railway / Heroku-style host** - push this folder to GitHub, create a Web Service, then:
- Build command: `pip install -r requirements.txt && python train.py`
- Start command: `gunicorn app:app` (the `Procfile` already says this)

**Docker**
```bash
docker build -t house-price-app .
docker run -p 8000:8000 house-price-app     # http://localhost:8000
```
The build re-trains the model so it always matches the installed scikit-learn version.

## API
```bash
curl -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" -d '{
  "OverallQual":7,"GrLivArea":1800,"TotalBsmtSF":1000,"GarageCars":2,"FullBath":2,
  "YearBuilt":2003,"LotArea":9500,"Fireplaces":1,"Neighborhood":"CollgCr","KitchenQual":"Gd"}'
```
Other endpoints: `GET /health`, `GET /api/model-info`.

## Results (20% hold-out test set, `random_state=42`, all 220 features)
| Model | MAE ($) | RMSE ($) | R2 |
|---|---|---|---|
| Lasso | 14,477 | 20,273 | 0.9256 |
| ElasticNet | 14,512 | 20,395 | 0.9247 |
| Gradient Boosting (tuned) | 14,261 | 20,463 | 0.9242 |
| Ridge | 14,933 | 21,086 | 0.9195 |
| Linear Regression | 14,768 | 21,139 | 0.9191 |
| Random Forest | 16,646 | 23,785 | 0.8976 |

Regularisation helps: Ridge/Lasso shrink the coefficients (largest |coef| 0.136 -> 0.044 / 0.075) and
Lasso removes 139 of the 220 features entirely, which reduces overfitting.

## About the deployed model
The web form asks for 10 features a user can realistically know (8 numeric + Neighborhood + Kitchen quality),
so the app model is trained on only those. Linear, Ridge, Lasso, ElasticNet and Gradient Boosting scored
the same in 5-fold CV (MAE about $17.5k), so Lasso was chosen as the simplest. Expect about +/-$17,800 error
on unseen houses.
