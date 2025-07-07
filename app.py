# 📄 app/app.py

import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="PricePulse - Stock Predictor", layout="centered")
st.title("📈 PricePulse - Stock Price Prediction")
st.markdown("Predict the next closing price of Netflix stock using an LSTM-based machine learning model.")

# Load dataset
try:
    st.write("🔍 Looking for data at: Data/Netflix_stock_data.csv")
    df = pd.read_csv('C:/Users/HP/OneDrive/Desktop/pricepulse/Data/Netflix_stock_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    st.success("✅ Data loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
    st.stop()

# Plot historical stock price
st.subheader("📊 Historical Closing Prices")
st.line_chart(df['Close'])

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])
time_step = 60

if len(scaled_data) < time_step:
    st.warning("⚠️ Not enough data for prediction.")
    st.stop()

X_input = scaled_data[-time_step:].reshape(1, time_step, 1)

# Load model and predict
try:
    st.write("🔄 Loading LSTM model...")
    model = load_model('C:/Users/HP/OneDrive/Desktop/pricepulse/models/lstm_model.h5')
    prediction = model.predict(X_input)
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    st.success("✅ Prediction successful!")
except Exception as e:
    st.error(f"❌ Error loading or predicting with model: {e}")
    st.stop()

# Display predicted price
st.subheader("📌 Predicted Next Day Closing Price")
st.success(f"₹ {predicted_price:.2f}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and LSTM. Dataset: Netflix historical stock prices.")
