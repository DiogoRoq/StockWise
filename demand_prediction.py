import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocess the input data for training or prediction.
    """
    # Make a copy of the input data to avoid modifying the original
    df = df.copy()

    # Drop the 'record_ID' column (irrelevant for training/prediction)
    if 'record_ID' in df.columns:
        df.drop('record_ID', axis=1, inplace=True)

    # Split 'week' column into 'day', 'month', 'year'
    if 'week' in df.columns:
        df[['day', 'month', 'year']] = df['week'].str.split('/', expand=True)
        df.drop('week', axis=1, inplace=True)  # Drop the original 'week' column

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['store_id', 'sku_id'], drop_first=True)

    return df

def train_model(df):
    # Preprocess the data
    df = preprocess_data(df)
    print("Columns after preprocessing (training):", df.columns.tolist())

    # Remove outliers (keeping values below the 99th percentile)
    df = df[df.units_sold < df.units_sold.quantile(0.99)]

    # Splitting data into features (X) and target variable (y)
    X = df.drop('units_sold', axis=1)  # Features (all columns except 'units_sold')
    y = df['units_sold']  # Target variable

    # Feature scaling
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)

    # Target scaling (if needed)
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    # Creating train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Defining parameters for model optimization
    param_grid = {
        'n_estimators': [10, 20],
        'min_samples_split': [2, 3]
    }

    # Creating and training the RandomForestRegressor model
    model = RandomForestRegressor(n_jobs=-1)
    grid_search = GridSearchCV(model, param_grid, verbose=2, cv=3)
    grid_search.fit(X_train, y_train)

    # Making predictions and calculating error
    y_pred = grid_search.best_estimator_.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    # Save the feature names AFTER preprocessing
    feature_names = X.columns.tolist()

    # Return the model, scalers, feature names, and RMSE
    return grid_search.best_estimator_, feature_scaler, target_scaler, feature_names, rmse


def predict_demand(model, stock_data, feature_scaler, target_scaler, feature_columns):

    # Preprocess the input data (same as during training)
    stock_data = preprocess_data(stock_data)

    # Ensure the input data has the same columns as the training data
    # Add missing columns (if any) and remove extra columns
    for col in feature_columns:
        if col not in stock_data.columns:
            stock_data[col] = 0  # Add missing columns with default value 0

    # Reorder columns to match the training data
    stock_data = stock_data[feature_columns]

    # Scale the features using the same scaler from training
    stock_data_scaled = feature_scaler.transform(stock_data)

    # Make predictions
    predictions = model.predict(stock_data_scaled)

    # Inverse transform the predictions if target scaling was used
    if target_scaler:
        predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1))

    # Round the predictions to the nearest integer
    predictions = predictions.round().astype(int)

    # Add the predictions to the original data
    stock_data['Predicted Demand'] = predictions

    # Extract the sku_id from the one-hot encoded columns
    sku_id_columns = [col for col in stock_data.columns if col.startswith('sku_id_')]
    sku_ids = [col.replace('sku_id_', '') for col in sku_id_columns]

    # Create a DataFrame with sku_id and predicted demand
    results = []
    for sku_id in sku_ids:
        sku_col = f'sku_id_{sku_id}'
        if sku_col in stock_data.columns:
            demand = stock_data.loc[stock_data[sku_col] == 1, 'Predicted Demand'].values
            if len(demand) > 0:
                results.append({'sku_id': sku_id, 'Predicted Demand': demand[0]})

    return pd.DataFrame(results)