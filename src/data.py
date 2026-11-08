from pathlib import Path
import pandas as pd, yfinance as yf

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def load(ticker="SPY", start="2005-01-01", refresh=False) -> pd.DataFrame:
    path = RAW / f"{ticker}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize(); df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype("float64")
    df.to_parquet(path)
    return df
