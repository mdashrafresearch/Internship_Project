"""
Train a used-car selling-price prediction model.

Usage:
    python train_model.py

Produces:
    model/car_price_model.joblib   -> trained sklearn Pipeline
    model/metadata.joblib          -> dropdown choices + metrics for the app
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils.preprocessing import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    basic_clean,
)

DATA_PATH = os.path.join("data", "Car_details_v3.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "car_price_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.joblib")

RANDOM_STATE = 42


def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df = basic_clean(df)

    # Drop rows without a target or with missing critical numeric fields
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN] > 0]

    # Duplicate rows are common in this dataset
    df = df.drop_duplicates()

    return df


def build_pipeline():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_COLUMNS),
        ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
    ])

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def main():
    print("Loading and cleaning data...")
    df = load_and_clean()
    print(f"  {len(df)} usable rows after cleaning")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("Training RandomForestRegressor...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"  MAE:  {mae:,.0f}")
    print(f"  RMSE: {rmse:,.0f}")
    print(f"  R2:   {r2:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "brands": sorted(X["brand"].unique().tolist()),
        "fuels": sorted(X["fuel"].unique().tolist()),
        "seller_types": sorted(X["seller_type"].unique().tolist()),
        "transmissions": sorted(X["transmission"].unique().tolist()),
        "owners": sorted(X["owner"].unique().tolist()),
        "year_min": int(X["year"].min()),
        "year_max": int(X["year"].max()),
        "km_driven_median": float(X["km_driven"].median()),
        "mileage_median": float(X["mileage"].median()),
        "engine_median": float(X["engine"].median()),
        "max_power_median": float(X["max_power"].median()),
        "torque_median": float(X["torque"].median()),
        "seats_mode": float(X["seats"].mode()[0]),
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "n_rows_trained": len(X_train),
    }
    joblib.dump(metadata, METADATA_PATH)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
