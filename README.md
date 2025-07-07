# 📈 PricePulse - Netflix Stock Price Prediction

PricePulse is a stock price prediction web application that uses an LSTM (Long Short-Term Memory) model to forecast the **next day closing price** of Netflix stock based on historical data. Built using **Python**, **Keras**, **Streamlit**, and **MySQL**, this project demonstrates a practical implementation of deep learning for time series forecasting

## 🚀 Features

- 📊 Visualizes historical Netflix stock prices
- 🧠 Trains an LSTM-based neural network on closing price data
- 🔎 Evaluates model accuracy with actual vs predicted graph
- 📌 Predicts the next day closing price
- 🌐 Web interface using Streamlit

## 📁 Folder Structure
PricePulse/
│
├── app/
│ └── app.py # Streamlit app file
│
├── models/
│ ├── train_model.py # Trains LSTM and saves model
│ ├── evaluate_model.py # Evaluates LSTM predictions
│ └── lstm_model.h5 # Saved LSTM model
│
├── notebooks/
│ ├── eda_and_preprocessing.ipynb # EDA and preprocessing notebook
│
├── Data/
│ └── Netflix_stock_data.csv # Raw CSV file
│
├── notebooks/data/
│ ├── X.npy # Preprocessed input data
│ └── y.npy # Preprocessed target data
│
├── requirements.txt # Required Python libraries
└── README.md # You're reading this!

## Model Overview
Model Type: LSTM (Long Short-Term Memory)
Input: Last 60 days of closing prices
Output: Predicted next day closing price
Loss Function: Mean Squared Error
Optimizer: Adam

📊 Sample Output
📉 Evaluation Graph: Actual vs Predicted prices
💹 Streamlit App: Displays predicted price and chart

📚 Dataset
Source: Yahoo Finance - Netflix Historical Stock Data
Format: CSV with Date, Open, High, Low, Close, Volume, Adj Close

🛠️ Technologies Used
Python 3.11
Keras & TensorFlow
Streamlit
NumPy, Pandas, Matplotlib
Scikit-learn
