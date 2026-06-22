<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=180&section=header&text=Trading%20AI%20Bot&fontSize=46&fontColor=fff&animation=twinkling&fontAlignY=36&desc=ML-Powered%20Cryptocurrency%20Price%20Prediction&descSize=16&descAlignY=58&descColor=a0c4ff" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge)]()

</div>

---

## 📌 Overview

A Python-based machine learning bot that analyzes cryptocurrency market data to predict short-term price movements and generate buy/sell signals. The bot uses a **deep neural network** trained on real-time OHLCV data fetched via the Yahoo Finance API to classify whether a coin's price will rise or fall over the next 5 hours.

---

## 🧠 How It Works

```
 Market Data (yfinance)
        ↓
 Feature Engineering
 [Open, High, Low, Close, Volume]
        ↓
 Preprocessing (StandardScaler)
        ↓
 Neural Network (TensorFlow/Keras)
 Dense(64, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
        ↓
 Binary Prediction: Price UP ↑ or DOWN ↓ in 5 hours
```

### Prediction Logic
- Pulls **60 days of hourly BTC-USD data** from Yahoo Finance
- Shifts the closing price by **5 intervals** to create the future price target
- Labels each row: `1` if price is predicted to rise, `0` if predicted to fall
- Trains a **binary classification neural network** using Adam optimizer and binary cross-entropy loss

---

## 📁 Project Structure

```
trading_ai_bot/
│
├── fetch_data.py                        # Pulls and caches market data via yfinance
├── prepare_features.py                  # Feature engineering & target labeling
├── indicators.py                        # Technical indicator calculations
├── train_model.py                       # Model training pipeline
├── predict.py                           # Generates buy/sell signals from trained model
├── ai_trading_bot_neural_networks.py    # Core neural network model (Keras/TensorFlow)
├── ai_bot.py                            # Linear classifier baseline model
└── data/                                # Cached datasets
```

---

## ⚙️ Model Architecture

```python
model = tf.keras.Sequential([
    Dense(64, activation='relu'),   # Input layer
    Dense(64, activation='relu'),   # Hidden layer 1
    Dense(32, activation='relu'),   # Hidden layer 2
    Dense(1,  activation='sigmoid') # Output: buy (1) or sell (0)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

| Parameter | Value |
|---|---|
| Training epochs | 30 |
| Batch size | 32 |
| Train/test split | 80% / 20% |
| Validation split | 10% of training data |
| Input features | Open, High, Low, Close, Volume |
| Prediction window | 5 hours ahead |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Data Collection** | `yfinance`, custom web scrapers |
| **Data Processing** | `pandas`, `NumPy`, `StandardScaler` |
| **ML Models** | `TensorFlow/Keras`, `scikit-learn` |
| **Visualization** | `Matplotlib` |
| **Language** | `Python 3` |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow yfinance pandas numpy matplotlib scikit-learn
```

### Run the neural network model
```bash
python ai_trading_bot_neural_networks.py
```

### Run the linear classifier baseline
```bash
python ai_bot.py
```

### Full pipeline
```bash
python fetch_data.py        # 1. Fetch market data
python prepare_features.py  # 2. Engineer features
python train_model.py       # 3. Train model
python predict.py           # 4. Generate signals
```

---

## 📊 Training Output

The model plots training vs. validation accuracy across 30 epochs to monitor overfitting and convergence:

```
Epoch 1/30  — loss: 0.6842 — accuracy: 0.5312 — val_accuracy: 0.5401
Epoch 15/30 — loss: 0.5901 — accuracy: 0.6214 — val_accuracy: 0.6089
Epoch 30/30 — loss: 0.5512 — accuracy: 0.6598 — val_accuracy: 0.6341
```

---

## ⚠️ Disclaimer

> This project is for **educational and research purposes only**. It is not financial advice. Cryptocurrency markets are highly volatile — never trade with funds you cannot afford to lose.

---

## 🗺️ Roadmap

- [x] Neural network binary classifier for BTC-USD
- [x] Linear classifier baseline
- [x] Modular pipeline (fetch → prepare → train → predict)
- [ ] Add RSI, MACD, Bollinger Band indicators
- [ ] Expand to meme coin data (Dogecoin, Shiba Inu, etc.)
- [ ] Backtesting framework
- [ ] Live paper trading integration

---

<div align="center">

**Built by [Nandan Pullakandam](https://github.com/NP-Code99)**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/nandan-pullakandam)
[![Portfolio](https://img.shields.io/badge/GitHub-171515?style=flat-square&logo=github&logoColor=white)](https://github.com/NP-Code99)

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=100&section=footer" width="100%"/>

</div>
