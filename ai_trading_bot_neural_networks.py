# With this program, I utilized a neural network to learn a data set and make accurate predictions on cryptocurrency data

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf 

df = yf.download("BTC-USD", period="60d", interval="1h")
df.dropna(inplace=True) # Removes all the NAN values from the dataset
df.head()

df.columns = ["_".join(col).strip() for col in df.columns.values]

df["Future Price"] = df["Close_BTC-USD"].shift(-5) # Shifts the Close data by 5 to store the future values
df["Target"] = (df["Future Price"] > df["Close_BTC-USD"]).astype(int) # Calculates the target column by comparing future price with current price

df.dropna(inplace=True)

# Initializing the features and labels
features = df[["Open_BTC-USD", "High_BTC-USD", "Low_BTC-USD", "Close_BTC-USD", "Volume_BTC-USD"]] # X values
labels = df["Target"] # Y Values

scaler = StandardScaler() # Preprocessing all of the values
features_scaled = scaler.fit_transform(features)

# Splitting the data up into different sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size = 0.2, random_state = 42)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1], )), # Input Layer
    tf.keras.layers.Dense(64, activation='relu'), # Hidden Layers 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Output Layer
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

# Testing the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Plotting the accuracy of the model
plt.plot(history.history['accuracy'], label = "Train Accuracy")
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training History")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()