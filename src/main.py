import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import yfinance as yf

sns.set()
tf.random.set_seed(1234)


def download_stock_data(symbol, period):
    return yf.download(symbol, period=period, interval="1d")


def preprocess_data(data):
    scaler = MinMaxScaler()
    close_prices = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_prices)
    return pd.DataFrame(scaled_data), scaler


def build_model(learning_rate, num_layers, size_layer, input_shape, output_size, dropout_rate):
    model = tf.keras.Sequential()
    for _ in range(num_layers):
        model.add(tf.keras.layers.LSTM(size_layer, return_sequences=True, input_shape=input_shape))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.LSTM(size_layer))
    model.add(tf.keras.layers.Dense(output_size))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model


def predict_and_plot_stock(symbol, period='1y', sim=5, future=30, epoch=100, timestamp=10):
    df = download_stock_data(symbol, period)
    df_log, scaler = preprocess_data(df)

    model = build_model(learning_rate=0.01, num_layers=1, size_layer=128, input_shape=(timestamp, 1), output_size=1,
                        dropout_rate=0.2)

    X_train = []
    y_train = []
    for i in range(timestamp, len(df_log)):
        X_train.append(df_log.values[i - timestamp:i, 0])
        y_train.append(df_log.values[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model.fit(X_train, y_train, epochs=epoch, batch_size=32)

    predictions = []
    current_batch = X_train[-1].reshape((1, timestamp, 1))
    for i in range(future):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.roll(current_batch, -1)
        current_batch[:, -1:, 0] = current_pred

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Prepare dates for plotting
    last_date = df.index[-1]
    prediction_dates = pd.date_range(start=last_date, periods=future + 1)[1:]

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label='Actual Prices')
    plt.plot(prediction_dates, predicted_prices, label='Predicted Prices', linestyle='--')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return predicted_prices, prediction_dates


# Example usage
predicted_prices, prediction_dates = predict_and_plot_stock('NG', period='1y', sim=5, future=30)


