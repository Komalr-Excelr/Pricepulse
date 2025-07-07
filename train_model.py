import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import os

# Load preprocessed data
X = np.load('../data/X.npy')
y = np.load('../data/y.npy')

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

model.save('lstm_model.h5')
print("✅ Model trained and saved successfully as 'lstm_model.h5'")
