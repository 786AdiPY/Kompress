"""Create a synthetic house-price dataset and a 'bring-your-own' trained
scikit-learn regressor artifact — a second, unrelated model to prove the
pipeline compresses any model (different framework + regression task).
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sqft = rng.uniform(500, 4000, args.n)
    bedrooms = rng.integers(1, 6, args.n)
    bathrooms = rng.integers(1, 4, args.n)
    age_years = rng.integers(0, 80, args.n)
    price = (150 * sqft + 12000 * bedrooms + 9000 * bathrooms
             - 800 * age_years + rng.normal(0, 15000, args.n))
    df = pd.DataFrame({"sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
                       "age_years": age_years, "price": price})

    split = int(0.8 * args.n)
    os.makedirs("data", exist_ok=True)
    df.iloc[:split].to_csv("data/house_train.csv", index=False)
    df.iloc[split:].to_csv("data/house_test.csv", index=False)

    feats = ["sqft", "bedrooms", "bathrooms", "age_years"]
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    model.fit(df.iloc[:split][feats].values, df.iloc[:split]["price"].values)

    os.makedirs("artifacts/house_price", exist_ok=True)
    with open("artifacts/house_price/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"house data: train={split} test={args.n - split}")
    print("artifact -> artifacts/house_price/model.pkl (sklearn RandomForestRegressor)")


if __name__ == "__main__":
    main()
