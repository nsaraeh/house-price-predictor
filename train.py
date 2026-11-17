"""
Train, compare and save the house-price models.

Part A  - Bonus 1: compare Linear Regression with Ridge, Lasso and ElasticNet
          (plus the tree models from the main assignment) on ALL features,
          using exactly the preprocessing from the notebook.
Part B  - Bonus 2: train the model that powers the web app. The web form only
          asks for 10 easy-to-know features, so we compare the same linear
          models on those 10 features, pick the best by cross-validation,
          refit it on all the data and save it to model/house_model.joblib.

Run:  python train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (ElasticNetCV, LassoCV, LinearRegression,
                                  RidgeCV)
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "train.csv"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

SEED = 42

# Features asked in the web form (the model is trained on exactly these)
NUM_FEATURES = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars",
                "FullBath", "YearBuilt", "LotArea", "Fireplaces"]
CAT_FEATURES = ["Neighborhood", "KitchenQual"]
APP_FEATURES = NUM_FEATURES + CAT_FEATURES


# --------------------------------------------------------------------------
# Data loading + the notebook's preprocessing
# --------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA).drop(columns=["Id"])
    # Two huge houses sold very cheaply: clear outliers (same as notebook)
    df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000))]
    return df.reset_index(drop=True)


def notebook_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values + one-hot encoding + engineered features (Tasks 1-2)."""
    df = df.copy()
    none_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
                 "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                 "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                 "BsmtFinType2", "MasVnrType"]
    for c in none_cols:
        df[c] = df[c].fillna("None")
    for c in ["GarageYrBlt", "MasVnrArea"]:
        df[c] = df[c].fillna(0)
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda s: s.fillna(s.median()))
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    df["MSSubClass"] = df["MSSubClass"].astype(str)
    numeric = df.select_dtypes(include="number").columns
    enc = pd.get_dummies(df, drop_first=True, dtype=int)
    rare = [c for c in enc.columns
            if c not in numeric and enc[c].sum() < 10]
    enc = enc.drop(columns=rare)

    enc["TotalSF"] = enc["TotalBsmtSF"] + enc["1stFlrSF"] + enc["2ndFlrSF"]
    enc["HouseAge"] = enc["YrSold"] - enc["YearBuilt"]
    enc["RemodAge"] = enc["YrSold"] - enc["YearRemodAdd"]
    enc["TotalBath"] = (enc["FullBath"] + 0.5 * enc["HalfBath"]
                        + enc["BsmtFullBath"] + 0.5 * enc["BsmtHalfBath"])
    enc["TotalPorchSF"] = (enc["OpenPorchSF"] + enc["EnclosedPorch"]
                           + enc["3SsnPorch"] + enc["ScreenPorch"]
                           + enc["WoodDeckSF"])
    enc["HasPool"] = (enc["PoolArea"] > 0).astype(int)
    enc["HasGarage"] = (enc["GarageArea"] > 0).astype(int)
    enc["Has2ndFloor"] = (enc["2ndFlrSF"] > 0).astype(int)
    enc["Remodeled"] = (enc["YearRemodAdd"] != enc["YearBuilt"]).astype(int)
    return enc


def dollar_metrics(name, y_true_log, y_pred_log) -> dict:
    actual, pred = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {
        "Model": name,
        "MAE ($)": mean_absolute_error(actual, pred),
        "RMSE ($)": np.sqrt(mean_squared_error(actual, pred)),
        "RMSE (log)": np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        "R2": r2_score(actual, pred),
    }


# --------------------------------------------------------------------------
# Part A - Bonus 1: Linear vs Ridge vs Lasso (all features)
# --------------------------------------------------------------------------
def part_a(raw: pd.DataFrame) -> dict:
    print("\n=== PART A: model comparison on ALL features ===")
    enc = notebook_preprocessing(raw)
    X = enc.drop(columns="SalePrice")
    y = np.log1p(enc["SalePrice"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=SEED)
    scaler = StandardScaler().fit(X_tr)           # fit on train only
    X_tr_s = pd.DataFrame(scaler.transform(X_tr), columns=X.columns)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X.columns)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": RidgeCV(alphas=np.logspace(-1, 3, 30)),
        "Lasso": LassoCV(alphas=np.logspace(-4, -1, 30), cv=5, max_iter=50000),
        "ElasticNet": ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95],
                                   alphas=np.logspace(-4, -1, 30), cv=5,
                                   max_iter=50000),
        "Random Forest": RandomForestRegressor(n_estimators=300,
                                               random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            learning_rate=0.05, max_depth=3, n_estimators=400,
            random_state=SEED),   # tuned params from your Grid Search
    }
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(X_tr_s, y_tr)
        rows.append(dollar_metrics(name, y_te, m.predict(X_te_s)))
        fitted[name] = m
    res = pd.DataFrame(rows).set_index("Model")
    res.loc["Gradient Boosting", "Note"] = "tuned (400 trees, lr 0.05, depth 3)"

    print(res.drop(columns="Note").round(4).sort_values("RMSE (log)"))
    ridge, lasso, enet = fitted["Ridge"], fitted["Lasso"], fitted["ElasticNet"]
    print(f"\nBest Ridge alpha : {ridge.alpha_:.3f}")
    print(f"Best Lasso alpha : {lasso.alpha_:.5f}  -> kept "
          f"{int((lasso.coef_ != 0).sum())} of {X.shape[1]} features")
    print(f"Best ElasticNet  : alpha={enet.alpha_:.5f}, l1_ratio={enet.l1_ratio_}"
          f"  -> kept {int((enet.coef_ != 0).sum())} features")

    # Coefficient size: shows what regularisation does
    coef = pd.DataFrame({
        "Linear Regression": fitted["Linear Regression"].coef_,
        "Ridge": ridge.coef_, "Lasso": lasso.coef_}, index=X.columns)
    print("\nLargest |coefficient|  Linear:",
          round(coef["Linear Regression"].abs().max(), 3),
          "| Ridge:", round(coef["Ridge"].abs().max(), 3),
          "| Lasso:", round(coef["Lasso"].abs().max(), 3))

    res.round(5).to_csv(REPORT_DIR / "model_comparison_all_features.csv")
    top = coef["Lasso"].reindex(coef["Lasso"].abs().sort_values(
        ascending=False).index).head(15)
    top.round(5).to_csv(REPORT_DIR / "lasso_top15_coefficients.csv",
                        header=["coefficient"])

    # ---- charts -----------------------------------------------------------
    order = res.sort_values("RMSE (log)").index
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    palette = {"Linear Regression": "#c44e52", "Ridge": "#4c72b0",
               "Lasso": "#55a868", "ElasticNet": "#8172b3",
               "Random Forest": "#ccb974", "Gradient Boosting": "#64b5cd"}
    for ax, metric in zip(axes, ["MAE ($)", "RMSE ($)", "R2"]):
        vals = res.loc[order, metric]
        ax.bar(range(len(vals)), vals, color=[palette[n] for n in order])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_title(metric)
        if metric == "R2":
            ax.set_ylim(vals.min() - 0.03, 1)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.3f}" if metric == "R2" else f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=8)
    plt.suptitle("Model comparison on the test set (all features)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    top.sort_values().plot(kind="barh", color="purple", ax=ax)
    ax.set_title("Top 15 Lasso coefficients (scaled features)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "lasso_top_coefficients.png", dpi=130)
    plt.close()

    return {"alpha_ridge": float(ridge.alpha_), "alpha_lasso": float(lasso.alpha_),
            "lasso_kept": int((lasso.coef_ != 0).sum()),
            "n_features": int(X.shape[1])}


# --------------------------------------------------------------------------
# Part B - Bonus 2: the model that goes into the web app
# --------------------------------------------------------------------------
def make_pipeline(regressor) -> TransformedTargetRegressor:
    """Raw form inputs in -> price in dollars out (log transform is built in)."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", pre), ("model", regressor)]),
        func=np.log1p, inverse_func=np.expm1)


def part_b(raw: pd.DataFrame, info_a: dict) -> None:
    print("\n=== PART B: model for the web app (10 form features) ===")
    X = raw[APP_FEATURES]
    y = raw["SalePrice"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=SEED)
    cv = KFold(5, shuffle=True, random_state=SEED)
    alphas = np.logspace(-3, 3, 40)
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": RidgeCV(alphas=alphas),
        "Lasso": LassoCV(alphas=np.logspace(-5, -1, 40), cv=5, max_iter=50000),
        "ElasticNet": ElasticNetCV(l1_ratio=[.1, .5, .9],
                                   alphas=np.logspace(-5, -1, 40), cv=5,
                                   max_iter=50000),
        "Gradient Boosting": GradientBoostingRegressor(
            learning_rate=0.05, max_depth=3, n_estimators=300,
            subsample=0.8, random_state=SEED),
    }
    rows, cv_scores = [], {}
    for name, reg in candidates.items():
        pipe = make_pipeline(reg)
        # 5-fold CV on the training split, scored with MAE in dollars
        scores = -cross_val_score(make_pipeline(reg), X_tr, y_tr, cv=cv,
                                  scoring="neg_mean_absolute_error")
        cv_scores[name] = scores.mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "Model": name,
            "CV MAE ($)": scores.mean(),
            "Test MAE ($)": mean_absolute_error(y_te, pred),
            "Test RMSE ($)": np.sqrt(mean_squared_error(y_te, pred)),
            "Test R2": r2_score(y_te, pred),
        })
    res = pd.DataFrame(rows).set_index("Model")
    print(res.round(4).sort_values("CV MAE ($)"))
    res.round(5).to_csv(REPORT_DIR / "model_comparison_app_features.csv")

    best_name = min(cv_scores, key=cv_scores.get)
    print(f"\nSelected for the app (lowest cross-validated MAE): {best_name}")

    final = make_pipeline(candidates[best_name])
    final.fit(X, y)                      # refit on all 1,458 houses

    # Realistic input ranges for the web form's validation
    limits = {c: [float(X[c].min()), float(X[c].max())] for c in NUM_FEATURES}
    limits["YearBuilt"] = [int(X["YearBuilt"].min()), 2026]
    meta = {
        "model_name": best_name,
        "features": APP_FEATURES,
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "neighborhoods": sorted(X["Neighborhood"].unique().tolist()),
        "kitchen_quality": ["Ex", "Gd", "TA", "Fa"],
        "limits": limits,
        "defaults": {c: float(X[c].median()) for c in NUM_FEATURES},
        "test_metrics": res.loc[best_name].round(4).to_dict(),
        "full_feature_comparison": info_a,
        "n_training_rows": int(len(X)),
    }
    joblib.dump(final, MODEL_DIR / "house_model.joblib", compress=3)
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved model/house_model.joblib and model/model_meta.json")


if __name__ == "__main__":
    raw = load_raw()
    print("Training rows:", len(raw))
    info = part_a(raw)
    part_b(raw, info)
