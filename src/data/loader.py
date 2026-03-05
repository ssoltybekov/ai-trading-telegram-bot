import yfinance as yf
import pandas as pd

def load_dataset(ticker: str, period: str = "10y") -> pd.DataFrame:
    print(f"Загружаем данные для {ticker}")
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)

    if df.empty:
        raise ValueError(f"Данные для тикера '{ticker}' не найдены")
    
    df.dropna(inplace=True)
    

    return df

