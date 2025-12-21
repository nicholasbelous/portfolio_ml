# src/features.py

import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data")
PRICES_PATH = DATA_DIR / "prices.csv"
OUTPUT_PATH = DATA_DIR / "features.csv"


def load_prices():
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    prices.sort_index(inplace=True)
    return prices


def compute_momentum(prices, window):
    """
    Compute simple returns over a rolling window.
    """
    return prices.pct_change(window)


def compute_volatility(prices, window):
    """
    Compute rolling realized volatility.
    """
    returns = prices.pct_change()
    return returns.rolling(window).std()


def build_features(prices):
    features = {}

    features["mom_1m"] = compute_momentum(prices, 21)
    features["mom_3m"] = compute_momentum(prices, 63)
    features["vol_20d"] = compute_volatility(prices, 20)

    # Stack into long format
    feature_dfs = []
    for name, df in features.items():
        stacked = df.stack().rename(name)
        feature_dfs.append(stacked)

    features_long = pd.concat(feature_dfs, axis=1)

    # Drop rows with any missing features
    features_long = features_long.dropna()

    return features_long


def main():
    prices = load_prices()
    features = build_features(prices)

    features.to_csv(OUTPUT_PATH)

    print("Saved features to", OUTPUT_PATH)
    print("Feature shape:", features.shape)
    print("Feature columns:", list(features.columns))
    print("Date range:", features.index.get_level_values(0).min(),
          "→", features.index.get_level_values(0).max())


if __name__ == "__main__":
    main()