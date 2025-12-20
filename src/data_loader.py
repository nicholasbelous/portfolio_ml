# src/data_loader.py

import pandas as pd
from pathlib import Path
from datetime import datetime
import pandas_datareader.data as web

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "TSLA", "JPM", "JNJ", "V",
    "PG", "UNH", "HD", "MA", "BAC",
    "XOM", "CVX", "KO", "PEP", "DIS"
]

START_DATE = datetime(2018, 1, 1)
END_DATE = datetime(2024, 1, 1)
OUTPUT_PATH = DATA_DIR / "prices.csv"


def download_prices_stooq(tickers, start, end):
    prices = {}

    for ticker in tickers:
        print(f"Downloading {ticker} from Stooq...")
        df = web.DataReader(ticker, "stooq", start, end)

        if df.empty:
            raise ValueError(f"No data for {ticker}")

        prices[ticker] = df["Close"]

    prices = pd.DataFrame(prices)
    prices.sort_index(inplace=True)
    return prices


def clean_prices(prices):
    prices = prices.dropna(how="all")
    prices = prices.ffill()
    prices = prices.dropna()
    return prices


def main():
    prices = download_prices_stooq(TICKERS, START_DATE, END_DATE)
    prices = clean_prices(prices)

    prices.to_csv(OUTPUT_PATH)

    print("\nSaved prices to", OUTPUT_PATH)
    print("Shape:", prices.shape)
    print("Missing values:", prices.isna().sum().sum())


if __name__ == "__main__":
    main()
