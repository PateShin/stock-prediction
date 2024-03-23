# Stock Price Prediction with LSTM

This project is focused on predicting stock prices using Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN) suitable for sequential data. We utilize TensorFlow for constructing the LSTM model and Yahoo Finance (yfinance) to fetch historical stock data. The project also involves data preprocessing with scikit-learn's MinMaxScaler for normalizing the input features and matplotlib for visualizing the predicted stock prices against actual prices.

## Features

- Fetch stock data directly from Yahoo Finance using `yfinance`.
- Normalize stock prices using MinMaxScaler for optimal LSTM model performance.
- Build and train LSTM models customizable with different hyperparameters including learning rate, number of layers, and dropout rate.
- Predict future stock prices and visualize the predictions against actual historical prices.

## Dependencies

To run this project, you will need:

- Python 3.6+
- TensorFlow 2.x
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn
- yfinance

You can install these dependencies using `pip`:


## Usage

1. **Download Stock Data**: Use `download_stock_data(symbol, period)` to fetch historical stock data. The `symbol` parameter refers to the ticker symbol of the stock, and `period` defines the timeframe for which data is fetched.

2. **Preprocess Data**: Normalize the stock data using `preprocess_data(data)` to prepare it for the LSTM model.

3. **Build Model**: Construct the LSTM model with `build_model(learning_rate, num_layers, size_layer, input_shape, output_size, dropout_rate)`. Customize the model architecture by adjusting its parameters.

4. **Predict and Plot Stock Prices**: Use `predict_and_plot_stock(symbol, period='1y', sim=5, future=30, epoch=100, timestamp=10)` to train the model with historical data and predict future stock prices. The function plots the actual vs. predicted prices for visualization.

### Example

```python
predicted_prices, prediction_dates = predict_and_plot_stock('AAPL', period='1y', future=30)
  
