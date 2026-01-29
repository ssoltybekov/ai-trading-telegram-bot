import yfinance as yf
import pandas as pd

def load_dataset(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="10y", interval="1d")

    df.dropna(inplace=True)
    
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df

