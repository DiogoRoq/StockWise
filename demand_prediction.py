import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


data = pd.read_csv('data.csv')  
data['Date'] = pd.to_datetime(data['Date']) 
data.set_index('Date', inplace=True)
target = data['SoldCount'].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(target.reshape(-1, 1))


def create_sequences(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 30
X, y = create_sequences(scaled_data, time_step)


X = X.reshape(X.shape[0], X.shape[1], 1)


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


model = tf.keras.Sequential()
model.add(tf.keras.LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(tf.keras.LSTM(50, return_sequences=False))
model.add(tf.keras.Dense(25))
model.add(tf.keras.Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))


predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(data):
    data['month'] = pd.to_datetime(data['date']).dt.month
    data['year'] = pd.to_datetime(data['date']).dt.year
    data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek

    features = ['month', 'year', 'day_of_week', 'other_features']
    target = 'sales'


    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    return model

def predict_demand(model, stock_data):
    stock_data['month'] = pd.to_datetime(stock_data['date']).dt.month
    stock_data['year'] = pd.to_datetime(stock_data['date']).dt.year
    stock_data['day_of_week'] = pd.to_datetime(stock_data['date']).dt.dayofweek

    features = ['month', 'year', 'day_of_week', 'other_features']
    stock_data['predicted_demand'] = model.predict(stock_data[features])
    return stock_data