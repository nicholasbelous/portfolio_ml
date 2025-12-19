import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "TSLA", "JPM", "JNJ", "V",
    "PG", "UNH", "HD", "MA", "BAC",
    "XOM", "CVX", "KO", "PEP", "DIS"
]

START_DATE = "2018-01-01"
END_DATE = "2024-01-01"

def download_prices(tickers, start, end):
    """
    Download adjusted close prices for given tickers.
    Returns a DataFrame indexed by date with tickers as columns.
    """
    price_data = {}

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True
        )

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        price_data[ticker] = df["Close"]

    prices = pd.DataFrame(price_data)
    prices.sort_index(inplace=True)

    return prices

def clean_prices(prices):
    """
    Basic cleaning:
    - Drop dates where all prices are missing
    - Forward-fill small gaps
    - Drop remaining NaNs
    """
    prices = prices.dropna(how="all")
    prices = prices.ffill()
    prices = prices.dropna()

    return prices


def main():
    prices = download_prices(TICKERS, START_DATE, END_DATE)
    prices = clean_prices(prices)

    output_path = DATA_DIR / "prices.csv"
    prices.to_csv(output_path)

    print(f"\nSaved cleaned prices to {output_path}")
    print(f"Shape: {prices.shape}")
    print(f"Date range: {prices.index.min()} → {prices.index.max()}")


if __name__ == "__main__":
    main()