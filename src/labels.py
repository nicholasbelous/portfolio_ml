# src/labels.py

import pandas as pd
from pathlib import Path


DATA_DIR = Path("data")
PRICES_PATH = DATA_DIR / "prices.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
OUTPUT_PATH = DATA_DIR / "labels.csv"


HORIZON_DAYS = 21  # ~1 trading month


def load_prices():
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    prices.sort_index(inplace=True)
    return prices


def load_features_index():
    """
    Load feature index so labels align exactly with available features.
    """
    features = pd.read_csv(
        FEATURES_PATH,
        index_col=[0, 1],
        parse_dates=[0]
    )
    return features.index


def compute_future_returns(prices, horizon):
    """
    Compute future returns over the given horizon.
    """
    future_prices = prices.shift(-horizon)
    future_returns = (future_prices / prices) - 1
    return future_returns


def build_labels(prices, feature_index):
    future_returns = compute_future_returns(prices, HORIZON_DAYS)

    # Stack to long format
    labels_long = future_returns.stack().rename("target_return")

    # Align labels to feature dates only
    labels_long = labels_long.reindex(feature_index)


    # Drop any remaining NaNs (near the end of the series)
    labels_long = labels_long.dropna()

    return labels_long.to_frame()


def main():
    prices = load_prices()
    feature_index = load_features_index()

    labels = build_labels(prices, feature_index)
    labels.to_csv(OUTPUT_PATH)

    print("Saved labels to", OUTPUT_PATH)
    print("Label shape:", labels.shape)
    print("Date range:",
          labels.index.get_level_values(0).min(),
          "→",
          labels.index.get_level_values(0).max())


if __name__ == "__main__":
    main()
