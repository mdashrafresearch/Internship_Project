"""
Shared preprocessing utilities for the Car Price Predictor.

These functions are used both at training time (train_model.py) and at
inference time (app.py) so the feature engineering logic never drifts
out of sync between the two.
"""

import re
import numpy as np
import pandas as pd

KGM_TO_NM = 9.80665

CURRENT_YEAR_DEFAULT = 2026  # fallback if system clock unavailable


def extract_brand(name: str) -> str:
    """Get the car brand (first token of the 'name' column)."""
    if pd.isna(name):
        return "Unknown"
    return str(name).strip().split(" ")[0]


def clean_numeric_suffix(value):
    """Strip a trailing unit like ' kmpl', ' CC', ' bhp' and return a float."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if match:
        return float(match.group())
    return np.nan


def parse_torque(value):
    """
    Torque strings arrive in wildly inconsistent formats, e.g.:
        '190Nm@ 2000rpm'
        '12.7@ 2,700(kgm@ rpm)'
        '22.4 kgm at 1750-2750rpm'
        '113.75nm@ 4000rpm'
    We extract the leading numeric torque value and convert kgm -> Nm
    so every row ends up in the same unit.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).lower().replace(",", "")

    match = re.search(r"[-+]?\d*\.?\d+", s)
    if not match:
        return np.nan
    num = float(match.group())

    # If "kgm" appears anywhere (either right after the number or in
    # parentheses further along), convert to Nm.
    if "kgm" in s:
        num = num * KGM_TO_NM

    return num


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw-CSV -> cleaned dataframe with engineered numeric features.
    Safe to call on a single-row dataframe (inference) or the full
    training set.
    """
    df = df.copy()

    df["brand"] = df["name"].apply(extract_brand)

    df["mileage"] = df["mileage"].apply(clean_numeric_suffix)
    df["engine"] = df["engine"].apply(clean_numeric_suffix)
    df["max_power"] = df["max_power"].apply(clean_numeric_suffix)
    df["torque"] = df["torque"].apply(parse_torque)

    if "seats" in df.columns:
        df["seats"] = pd.to_numeric(df["seats"], errors="coerce")

    df = df.drop(columns=["name"], errors="ignore")

    return df


FEATURE_COLUMNS = [
    "brand",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
]

CATEGORICAL_COLUMNS = ["brand", "fuel", "seller_type", "transmission", "owner"]
NUMERIC_COLUMNS = ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats"]

TARGET_COLUMN = "selling_price"
