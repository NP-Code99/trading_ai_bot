# In this file, I will be using the data to make accurate predictions of the stock market

import pandas as pd
import joblib
from features.prepare_features import load_and_prepare

def predict(filepath, model_path="model/xgb_model.pkl", return_probs=False):
    """
    Loads model and predicts target (pump or not) on new data.

    Args:
        filepath (str): Path to input CSV (enriched with indicators)
        model_path (str): Path to trained model file
        return_probs (bool): If True, return probabilities instead of binary labels

    Returns:
        pd.DataFrame: Data with predictions
    """
    # Load data
    df = load_and_prepare(filepath)

    # Load model
    model = joblib.load(model_path)

    # Match training features (exclude time/labels)
    features = df.columns.difference(['timestamp'])

    # Make predictions
    if return_probs:
        df['prob_pump'] = model.predict_proba(df[features])[:, 1]  # Probability of class 1
    else:
        df['prediction'] = model.predict(df[features])

    return df


if __name__ == "__main__":
    # EXAMPLE USAGE
    input_csv = "data/kucoin_pepe_usdt_hourly.csv"
    result = predict(input_csv, return_probs=True)
    print(result[['timestamp', 'close', 'prob_pump']].tail())