# Stock Price Prediction using LSTM

This project aims to predict the future prices of stocks using Long Short-Term Memory (LSTM) networks, a type of recurrent neural network suitable for time series forecasting. The TensorFlow and Keras libraries are utilized to build the model, while the stock data is fetched using the Yahoo Finance API through the `yfinance` and `pandas_datareader` libraries.

## Features

- **Data Fetching**: Utilizes `yfinance` to download historical stock data.
- **Preprocessing**: Employs `MinMaxScaler` from `sklearn` for normalizing the stock prices.
- **Model**: Builds an LSTM-based model using TensorFlow and Keras for predicting future stock prices.
- **Visualization**: Plots the predicted stock prices against the actual prices, including prediction intervals.

## Prerequisites

Before you can run this project, you need to have the following installed:

- Python 3.6 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn
- `yfinance` and `pandas_datareader` for fetching stock data

## Installation

To install the required libraries, run the following command:

```bash
pip install numpy matplotlib seaborn pandas scikit-learn tensorflow yfinance pandas_datareader
```

## Usage
`predict_stock(Symbol, period, future_days)`
Example: 
```
predict_stock('NG', '1y', 30)
```
# Example: Predicting the stock prices for 'NG' over a period of 1 year with a 30-day future prediction
predict_stock('NG', '1y', 30)
Replace 'your_script_name' with the name of the Python script file where the predict_stock function is defined.


# Visualization
The output graph displays the actual stock prices in black, the forecasted prices in blue, and the prediction interval shaded in between, providing a visual assessment of the model's accuracy and uncertainty.
