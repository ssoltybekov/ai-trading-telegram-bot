import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model(df: pd.DataFrame, model_path: str = "models/model.pkl") -> None:
    feature_cols = ["return", "SMA_5", "SMA_10", "volatility_5"]
    X = df[feature_cols]
    y = df["target"]

    split_idx = int(len(df)*0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, model_path)
    print(f"Модель сохранена: {model_path}")




    

