import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from tqdm import tqdm
from scipy.stats import norm

from pandas_datareader import data as pdr
import yfinance as yf

sns.set()
tf.random.set_seed(1234)

yf.pdr_override()

num_layers = 1
size_layer = 128
timestamp = 10
epoch = 100
dropout_rate = 0.8
test_size = 30
learning_rate = 0.01


def predict_stock(symbol, period, future):
    test_size = future

    # Download dataframe using pandas_datareader
    df = pdr.get_data_yahoo(symbol, period=period, interval="1d")
    df.to_csv('data.csv')
    df = pd.read_csv('data.csv')

    minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32'))  # Close index
    df_log = minmax.transform(df.iloc[:, 4:5].astype('float32'))  # Close index
    df_log = pd.DataFrame(df_log)

    df_train = df_log

    class Model(tf.keras.Model):
        def __init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias=0.1):
            super(Model, self).__init__()

            self.lstm_layers = [tf.keras.layers.LSTM(size_layer, return_sequences=True) for _ in range(num_layers)]
            self.dropout = tf.keras.layers.Dropout(forget_bias)
            self.dense = tf.keras.layers.Dense(output_size)

            self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        def call(self, inputs):
            x = inputs
            for lstm_layer in self.lstm_layers:
                x = lstm_layer(x)
            x = self.dropout(x)
            outputs = self.dense(x[:, -1, :])
            return outputs

    def calculate_mse(real, predict):
        real = np.array(real)
        predict = np.array(predict)
        mse = np.mean(np.square(real - predict))
        return mse

    def anchor(signal, weight):
        buffer = []
        last = signal[0]
        for i in signal:
            smoothed_val = last * weight + (1 - weight) * i
            buffer.append(smoothed_val)
            last = smoothed_val
        return buffer

    def forecast():
        modelnn = Model(learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate)
        date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

        pbar = tqdm(range(epoch), desc='train loop')
        for i in pbar:
            total_loss = []
            for k in range(0, df_train.shape[0] - 1, timestamp):
                index = min(k + timestamp, df_train.shape[0] - 1)
                batch_x = np.expand_dims(df_train.iloc[k: index, :].values, axis=0)
                batch_y = df_train.iloc[k + 1: index + 1, :].values

                with tf.GradientTape() as tape:
                    logits = modelnn(batch_x)
                    loss = tf.reduce_mean(tf.square(batch_y - logits))  # MSE loss

                gradients = tape.gradient(loss, modelnn.trainable_variables)
                modelnn.optimizer.apply_gradients(zip(gradients, modelnn.trainable_variables))

                total_loss.append(loss)

            pbar.set_postfix(cost=np.mean(total_loss))

        future_day = test_size

        output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
        output_predict[0] = df_train.iloc[0]
        upper_b = (df_train.shape[0] // timestamp) * timestamp

        for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
            out_logits = modelnn(np.expand_dims(df_train.iloc[k: k + timestamp], axis=0))
            output_predict[k + 1: k + timestamp + 1] = out_logits

        if upper_b != df_train.shape[0]:
            out_logits = modelnn(np.expand_dims(df_train.iloc[upper_b:], axis=0))
            output_predict[upper_b + 1: df_train.shape[0] + 1] = out_logits
            future_day -= 1
            date_ori.append(date_ori[-1] + timedelta(days=1))

        for i in range(future_day):
            o = output_predict[-future_day - timestamp + i:-future_day + i]
            out_logits = modelnn(np.expand_dims(o, axis=0))
            output_predict[-future_day + i] = out_logits[-1]
            date_ori.append(date_ori[-1] + timedelta(days=1))

        output_predict = minmax.inverse_transform(output_predict)
        deep_future = anchor(output_predict[:, 0], 0.4)

        return deep_future

    results = forecast()  # Run forecast once

    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
    for i in range(test_size):
        date_ori.append(date_ori[-1] + timedelta(days=1))
    date_ori = pd.Series(date_ori).dt.strftime(date_format='%Y-%m-%d').tolist()

    mse = calculate_mse(df['Close'].values, results[:-test_size])

    # Calculate prediction intervals (assuming normally distributed errors)
    std_dev = np.sqrt(mse)
    lower_bound = results - 1.96 * std_dev  # 95% confidence interval
    upper_bound = results + 1.96 * std_dev

    plt.figure(figsize=(15, 7))
    plt.plot(date_ori, results, label='Forecast')
    plt.plot(df.iloc[:, 0].tolist(), df['Close'], label='Actual Trend', color='black', linewidth=2)
    plt.fill_between(date_ori, lower_bound, upper_bound, alpha=0.3, label='Prediction Interval')
    plt.title(f'Stock: {symbol} MSE: {mse:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

predict_stock('NG', '1y', 30)
