import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import configparser
import os
import joblib  # or you can use `import pickle`
import pickle
from flask import Flask, jsonify, sendfile, Response
import io
import requests

response = requests.get("http://data-preprocessing-service:80/export_processed_data")

#Example with models
response = requests.get(url)
if response.status_code == 200:
    # Write the content to a file
    with open('processed_data.csv', 'w') as f:  # Use 'wb' for binary write mode
        f.write(response.content)
else:
    print(f"Failed to download file. Status code: {response.status_code}")


def load_config(file_path):
    """Load configuration from file."""
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def data_preparation(df):
    """Prepare data by handling missing values and splitting into features and target."""
    if df.isnull().sum().any():
        df = df.dropna()  # simplistic approach; you can impute or handle missing values as needed

    y = df['AdoptionSpeed']
    X = df.drop(['AdoptionSpeed'], axis=1)
    return X, y

def train_validation_test_split(X, y, test_size, validation_size):
    """Split data into train, validation, and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, y_train, model_type, params=None):
    """Train the model using the specified model type and parameters."""
    if model_type == 'XGBoost':
        model = XGBClassifier(**params)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(**params)
    else:
        raise ValueError("Invalid model type. Choose 'XGBoost' or 'RandomForest'.")
    model.fit(X_train, y_train)
    return model

def main():
    # Load configuration
    config = load_config('parameter.env')
    file = config.get('PARAMETERS', 'PROCESSED_FILE')
    model_type = config.get('PARAMETERS', 'SAVED_MODELS_PATH')
    max_depth = int(config.get('PARAMETERS', 'MAX_DEPTH'))
    n_estimators = int(config.get('PARAMETERS', 'N_ESTIMATORS'))
    test_size = float(config.get('PARAMETERS', 'TEST_SIZE'))
    validation_size = float(config.get('PARAMETERS', 'VALIDATION_SIZE'))

    # Load and prepare data
    print("Reading file from " + file)
    data = load_data(file)
    X, y = data_preparation(data)

    # Split data into training, validation, and testing sets
    print("Data Modelling Stage1: train_validation_test_split")
    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(X, y, test_size, validation_size)

    # Save train, validation, and test sets to CSV
    print("Saving datasets to CSV files...")
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    print("Datasets saved successfully!")

    # Train and save the XGBoost model
    print("Training XGBoost model...")
    xgb_params = {'max_depth': max_depth, 'n_estimators': n_estimators}
    xgb_model = train_model(X_train, y_train, 'XGBoost', xgb_params)
    joblib.dump(xgb_model, 'xgb_model.pkl')
    print("XGBoost model saved as 'xgb_model.pkl'.")

    # Train and save the RandomForest model
    print("Training RandomForest model...")
    rf_params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'random_state': 42}
    rf_model = train_model(X_train, y_train, 'RandomForest', rf_params)
    joblib.dump(rf_model, 'rf_model.pkl')
    print("RandomForest model saved as 'rf_model.pkl'.")

    return X_train, X_test, y_train, y_test, xgb_model, rf_model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, xgb_model, rf_model =main()




app = Flask(__name__)

@app.route('/export_models', methods=['GET'])
def export_models():
    # Assume model, Xtrain, Xtest, y_train, y_test are already trained/created
    model_data = {
        'xgb_model': xgb_model,
        'rf_model': rf_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    serialized_data = pickle.dumps(model_data)
    return Response(serialized_data, content_type='application/octet-stream')


if __name__ == "__main__":  
    app.run(host = '0.0.0.0', port = 8080)