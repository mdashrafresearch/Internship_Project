# 🚗 Car Price Predictor

A machine learning web app that predicts the fair resale price of a used car
from its listing details (brand, year, kilometers driven, fuel type, engine
specs, etc.). Built with **scikit-learn** and **Streamlit**.

Trained on the `Car_details_v3` dataset (~8,100 real used-car listings).

**Model performance (held-out test set):**
- R² score: **0.925**
- Mean Absolute Error: **≈ ₹72,000**

## Project structure

```
car-price-predictor/
├── app.py                     # Streamlit app
├── train_model.py             # Training script
├── requirements.txt
├── data/
│   └── Car_details_v3.csv     # Training data
├── model/
│   ├── car_price_model.joblib # Trained sklearn Pipeline
│   └── metadata.joblib        # Dropdown options + metrics used by the app
└── utils/
    └── preprocessing.py       # Shared feature-engineering logic
```

## How it works

1. **Preprocessing** (`utils/preprocessing.py`) cleans the raw CSV:
   - Extracts `brand` from the free-text `name` column
   - Strips units from `mileage` ("23.4 kmpl" → `23.4`), `engine` ("1248 CC" → `1248`), and `max_power` ("74 bhp" → `74`)
   - Parses the inconsistent `torque` column (handles `Nm`, `nm`, and `kgm` formats, converting kgm → Nm) into a single numeric value
2. **Model**: a scikit-learn `Pipeline` with a `ColumnTransformer` (median-impute + one-hot encode categoricals) feeding a `RandomForestRegressor`.
3. **App**: `app.py` loads the saved pipeline and lets a user fill in a form to get an instant price estimate.

## Run locally

```bash
git clone https://github.com/<your-username>/car-price-predictor.git
cd car-price-predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (Optional) retrain the model from scratch
python train_model.py

# Launch the app
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Publish to GitHub

From inside the `car-price-predictor` folder:

```bash
git init
git add .
git commit -m "Initial commit: car price predictor model + Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/car-price-predictor.git
git push -u origin main
```

> Create the empty repo first at https://github.com/new (don't initialize it
> with a README, so the push above doesn't conflict).

## Deploy on Streamlit Community Cloud (free)

1. Push the project to GitHub (steps above).
2. Go to **https://share.streamlit.io** and sign in with your GitHub account.
3. Click **"New app"**.
4. Select your repository, the `main` branch, and set the main file path to
   `app.py`.
5. Click **"Deploy"**. Streamlit Cloud installs `requirements.txt` and
   launches the app — you'll get a public URL like
   `https://<your-app-name>.streamlit.app`.
6. Any future `git push` to `main` automatically redeploys the app.

## Retraining on new data

Replace `data/Car_details_v3.csv` with an updated CSV using the same column
schema, then run:

```bash
python train_model.py
```

This overwrites `model/car_price_model.joblib` and `model/metadata.joblib`.
Commit and push those files to update the deployed app.

## Tech stack

- Python, pandas, scikit-learn (RandomForestRegressor)
- Streamlit (UI + hosting)
- joblib (model serialization)
"# Internship_Project" 
