import pandas as pd
import joblib

def make_prediction(df: pd.DataFrame, model_path: str = "models/model.pkl") -> dict:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]

    feature_cols = ["return", "SMA_5", "SMA_10", "volatility_5"]
    X = df[feature_cols].iloc[[-1]]

    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    return {
    "signal": "UP 📈" if prediction == 1 else "DOWN 📉",
    "confidence": f"{max(probability):.0%}",
    "price": round(float(df["Close"].iloc[-1]), 2),
    "date": str(df.index[-1].date())
    }