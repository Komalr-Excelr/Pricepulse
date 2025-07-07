import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import math
import os

# Set paths (adjust if needed)
x_path = 'C:/Users/HP/OneDrive/Desktop/pricepulse/notebooks/data/X.npy'
y_path = 'C:/Users/HP/OneDrive/Desktop/pricepulse/notebooks/data/y.npy'
model_path = 'C:/Users/HP/OneDrive/Desktop/pricepulse/models/lstm_model.h5'

# Check all files exist
if not all(os.path.exists(p) for p in [x_path, y_path, model_path]):
    print("❌ One or more files are missing.")
    print(f"Check these paths:\n- {x_path}\n- {y_path}\n- {model_path}")
    exit()

# Load data
X = np.load(x_path)
y = np.load(y_path)

# Split into train/test
train_size = int(len(X) * 0.8)
X_test = X[train_size:]
y_test = y[train_size:]

# Load model
model = load_model(model_path)

# Predict
predictions = model.predict(X_test)

# Reshape y for scaler
y_all = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
predictions = predictions.reshape(-1, 1)

# Scale back to original prices
scaler = MinMaxScaler()
scaler.fit(y_all)

predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test)

# Evaluate
rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.2f}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title("📉 LSTM Model - Actual vs Predicted Stock Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
