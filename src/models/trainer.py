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

    



    

