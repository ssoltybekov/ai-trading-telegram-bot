import pandas as pd

def createFeatures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return"] = df["Close"].pct_change()

    df["SMA_5"] =df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()

    df["volatility_5"] = df["Close"].rolling(5).std()
    df["volatility_10"] = df["Close"].rolling(10).std()

    df.dropna(inplace=True)

    return df