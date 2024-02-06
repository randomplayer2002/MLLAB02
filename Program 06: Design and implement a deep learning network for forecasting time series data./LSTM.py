import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
# from math import sqrt

data = pd.read_csv('Electric_Production.csv')
time_series = data['value'].values.reshape(-1, 1)
scaler = MinMaxScaler()
time_series_normalized = scaler.fit_transform(time_series)
# print(time_series_normalized)

train_size = int(len(time_series_normalized) * 0.8)
train_data = time_series_normalized[:train_size]
test_data = time_series_normalized[train_size:]

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

model = Sequential()
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=64)

y_pred = model.predict(X_test)
y_pred_original_scale = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_original_scale = scaler.inverse_transform(y_test.reshape(-1, 1))

# rmse = sqrt(mean_squared_error(y_test_original_scale, y_pred_original_scale))
# print(f"Root Mean Squared Error: {rmse}")

plt.figure(figsize=(12, 6))
plt.plot(y_test_original_scale, label='Actual')
plt.plot(y_pred_original_scale, label='Predict')
plt.legend()
plt.title('Time Series Forecasting with LSTM')
plt.xlabel('time')
plt.ylabel('value')
plt.show()
