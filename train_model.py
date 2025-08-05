# In this file, I will be training and testing all of the data

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

from features.prepare_features import load_and_prepare

def create_labels(df, threshold=0.05, future_window=6):
    """
    Adds a target column: 1 if price will rise > threshold in future_window hours, else 0.
    """

    df = df.copy()
    df['future_return'] = df['close'].shift(-future_window) / df['close'] - 1
    df['target'] = (df['future_return'] > threshold).astype(int)
    df.dropna(inplace=True)
    return df

def train_xgboost(df, model_path="model/xgb_model.pkl"):
    # Define feature columns
    feature_cols = df.columns.difference(['timestamp', 'future_return', 'target'])

    X = df[feature_cols]
    Y = df['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, Y_train)

    # Evaluate
    Y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred))