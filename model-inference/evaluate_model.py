import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import joblib
import requests
import pickle  # or another appropriate library depending on the model format
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch the model and data from the model training service
response = requests.get("http://model-training-service:80/export_models")

if response.status_code == 200:
    # Load the received data
    data = pickle.loads(response.content)
    model = data['model']
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    print("Model and data received successfully")
else:
    print("Failed to receive the model and data")
    # Handle the error (retry, log, etc.)
    exit()  # Exit if model and data are not received

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print the classification report, confusion matrix, and accuracy."""
    predictions = model.predict(X_test)

    # Print accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Decision on model performance
    print("Model performance analysis:")
    if accuracy > 0.4:  # Example threshold, adjust as needed
        print("Model performance is good based on accuracy. Proceeding with the trained model.")
    else:
        print("Model performance is not satisfactory based on accuracy. Further analysis or model improvement may be needed.")
    print("Detailed metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    report = classification_report(y_test, predictions, output_dict=True)
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-Score: {report['weighted avg']['f1-score']:.4f}")

    # Return the evaluation results for further use
    return {
        "accuracy": accuracy,
        "classification_report": report,
        # Include other metrics as needed
    }

def main():
    # Evaluate the model
    print("Data Modelling Stage4: output result")
    evaluate_model(model, X_test, y_test)

    # Hyperparameter tuning (optional)
    print("Data Modelling Stage5: Hyperparameter Tuning")
    if model_type == 'XGBoost':
        model = XGBClassifier()
        param_grid = {'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200]}
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200]}
    else:
        raise ValueError("Unknown model_type specified")
    
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    # Evaluate the best model
    best_model = grid_search.best_estimator_
    print("Evaluating the best model after hyperparameter tuning:")
    evaluate_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
