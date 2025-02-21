import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error

def create_sequences(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step, -1])  # Target variable is the last column
    return np.array(X), np.array(y)

def train_model(data):
    # Convert dates if there are any date columns
    if 'ReleaseYear' in data.columns:
        data['ReleaseYear'] = data['ReleaseYear'].astype(str)
    
    # Feature engineering
    data['StrengthFactor'] = data['StrengthFactor'].astype(float)
    data['PriceReg'] = data['PriceReg'].astype(float)
    data['LowUserPrice'] = data['LowUserPrice'].astype(float)
    data['LowNetPrice'] = data['LowNetPrice'].astype(float)
    
    features = ['StrengthFactor', 'PriceReg', 'LowUserPrice', 'LowNetPrice', 'ReleaseYear']
    target = 'SoldCount'

    # Scaling the features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(data[features])
    scaled_target = target_scaler.fit_transform(data[[target]])

    # Combining scaled features and target
    scaled_data = np.hstack((scaled_features, scaled_target))

    # Creating sequences for LSTM
    time_step = 30
    X, y = create_sequences(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], len(features) + 1)

    # Splitting the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Building the LSTM model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, len(features) + 1)))
    model.add(tf.keras.layers.LSTM(50, return_sequences=False))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

    # Making predictions
    predictions = model.predict(X_test)
    predictions = target_scaler.inverse_transform(predictions)

    mse = mean_squared_error(target_scaler.inverse_transform(y_test.reshape(-1, 1)), predictions)
    print(f'Mean Squared Error: {mse}')

    return model, feature_scaler, target_scaler

def predict_demand(model, stock_data, feature_scaler, target_scaler):
    # Convert dates if there are any date columns
    if 'ReleaseYear' in stock_data.columns:
        stock_data['ReleaseYear'] = stock_data['ReleaseYear'].astype(str)
    
    # Feature engineering
    stock_data['StrengthFactor'] = stock_data['StrengthFactor'].astype(float)
    stock_data['PriceReg'] = stock_data['PriceReg'].astype(float)
    stock_data['LowUserPrice'] = stock_data['LowUserPrice'].astype(float)
    stock_data['LowNetPrice'] = stock_data['LowNetPrice'].astype(float)
    
    features = ['StrengthFactor', 'PriceReg', 'LowUserPrice', 'LowNetPrice', 'ReleaseYear']

    # Scaling the features
    scaled_features = feature_scaler.transform(stock_data[features])

    # Combining scaled features and initializing target as zeros for consistency
    scaled_data = np.hstack((scaled_features, np.zeros((scaled_features.shape[0], 1))))

    # Creating sequences for LSTM
    time_step = 30
    X, _ = create_sequences(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], len(features) + 1)

    # Making predictions
    predictions = model.predict(X)
    
    # Inverse transform the predictions to get them back to the original scale
    predictions = target_scaler.inverse_transform(predictions)

    # Ensure the predictions are non-negative
    predictions = np.maximum(predictions, 0)

    # Adjust the length of stock_data to match the predictions
    stock_data = stock_data.iloc[time_step:]
    stock_data['predicted_demand'] = predictions
    return stock_data