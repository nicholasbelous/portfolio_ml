import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# Config
TRAIN_YEARS = 3
TRADING_DAYS = 252
RIDGE_ALPHA = 1.0

FEATURE_PATH = "data/features.csv"
LABEL_PATH = "data/labels.csv"
OUTPUT_PATH = "data/predictions.csv"


# Load data
def load_data():
    X = pd.read_csv(FEATURE_PATH, index_col=[0, 1], parse_dates=[0])
    y = pd.read_csv(LABEL_PATH, index_col=[0, 1], parse_dates=[0])
    return X, y


# Training
def rolling_train_predict(X, y):
    dates = sorted(y.index.get_level_values(0).unique())
    tickers = y.index.get_level_values(1).unique()

    predictions = []

    window = TRAIN_YEARS * TRADING_DAYS

    for i in range(window, len(dates)):
        train_dates = dates[i - window:i]
        test_date = dates[i]

        # Training set
        X_train = X.loc[train_dates]
        y_train = y.loc[train_dates]

        # Test set (features only)
        X_test = X.loc[test_date]

        # Standardize features (fit only on train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_train_scaled, y_train.values.ravel())

        # Predict
        preds = model.predict(X_test_scaled)

        preds_df = pd.DataFrame(
            preds,
            index=X_test.index,
            columns=["predicted_return"]
        )

        predictions.append(preds_df)

    return pd.concat(predictions)


# Main
def main():
    X, y = load_data()
    preds = rolling_train_predict(X, y)

    preds.to_csv(OUTPUT_PATH)

    print("Saved predictions to", OUTPUT_PATH)
    print("Prediction shape:", preds.shape)
    print(preds.head())


if __name__ == "__main__":
    main()
