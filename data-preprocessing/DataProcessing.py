import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import configparser
import os


def read_config(config_file):
    """
    Read configuration from the given file.

    Parameters:
    config_file (str): Path to the configuration file.

    Returns:
    dict: Dictionary with configuration parameters.
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    params = {
        'input_file': config.get('PARAMETERS', 'INPUT_FILE'),
        'output_file': config.get('PARAMETERS', 'PROCESSED_FILE'),
        'cols_to_use': config.get('PARAMETERS', 'COLS_TO_USE').split(','),
        'cat_cols': config.get('PARAMETERS', 'CAT_COLS').split(','),
        'float_cols': config.get('PARAMETERS', 'FLOAT_COLS').split(','),
    }
    return params


def read_data(input_file):
    """
    Read data from the given CSV file.

    Parameters:
    input_file (str): Path to the input file.

    Returns:
    DataFrame: Loaded data.

    Raises:
    FileNotFoundError: If the input file does not exist.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")
    return pd.read_csv(input_file)


def feature_engineering(data):
    """
    Add new features to the data.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    DataFrame: Data with new features.
    """
    data['Free'] = data['Fee'].apply(lambda x: 1 if x == 0 else 0)
    data['NameorNO'] = data['NameorNO'].apply(lambda x: 1 if x == 'Y' else 0)
    data['BreedPure'] = data['BreedPure'].apply(lambda x: 1 if x == 'Y' else 0)
    return data


def select_columns(data, cols_to_use):
    """
    Select specified columns from the data.

    Parameters:
    data (DataFrame): Input data.
    cols_to_use (list): List of columns to use.

    Returns:
    DataFrame: Data with selected columns.
    """
    return data[[col for col in cols_to_use if col in data.columns]]


def one_hot_encode(data, cat_cols):
    """
    Apply one-hot encoding to categorical columns.

    Parameters:
    data (DataFrame): Input data.
    cat_cols (list): List of categorical columns.

    Returns:
    DataFrame: Data with one-hot encoded columns.
    """
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(data[cat_cols])
    ohe_data = pd.DataFrame(
        ohe.transform(data[cat_cols]),
        columns=ohe.get_feature_names_out(cat_cols),
        index=data.index
    )
    return pd.concat([data, ohe_data], axis=1).drop(columns=cat_cols)


def scale_features(data, float_cols):
    """
    Scale numerical features using MinMaxScaler.

    Parameters:
    data (DataFrame): Input data.
    float_cols (list): List of numerical columns.

    Returns:
    DataFrame: Data with scaled features.
    """
    scaler = MinMaxScaler()
    for col in float_cols:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data


def save_data(data, output_file):
    """
    Save the processed data to a CSV file.

    Parameters:
    data (DataFrame): Data to save.
    output_file (str): Path to the output file.
    """
    data.to_csv(output_file, index=False)


def handle_missing_values(data):
    """
    Handle missing values in the data.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    DataFrame: Data with missing values handled.
    """
    return data.dropna()


def handle_outliers(data, float_cols):
    """
    Handle outliers in numerical columns by capping them at the 1st and 99th percentiles.

    Parameters:
    data (DataFrame): Input data.
    float_cols (list): List of numerical columns.

    Returns:
    DataFrame: Data with outliers handled.
    """
    for col in float_cols:
        lower_limit = data[col].quantile(0.01)
        upper_limit = data[col].quantile(0.99)
        data[col] = data[col].clip(lower=lower_limit, upper=upper_limit)
    return data


def clean_column_names(data):
    """
    Clean the column names to remove special characters.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    DataFrame: Data with cleaned column names.
    """
    cleaned_col_names = []
    for col in data.columns:
        clean_name = col.replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('   ', '').replace(
            '  ', '').replace(' ', '')
        cleaned_col_names.append(clean_name)
    data.columns = cleaned_col_names
    return data


def main():
    """
    Main function to read configuration, process data, and save the processed data.
    """
    config_file = 'parameter.env'
    try:
        # Read configuration
        params = read_config(config_file)
        input_file = params['input_file']
        output_file = params['output_file']
        cols_to_use = params['cols_to_use']
        cat_cols = params['cat_cols']
        float_cols = params['float_cols']

        print(f"Reading file from {input_file}")

        # Read data
        pets_data = read_data(input_file)

        # Feature engineering
        print("Data Processing Stage1: feature engineering")
        pets_data = feature_engineering(pets_data)

        # Select columns
        pets_data = select_columns(pets_data, cols_to_use)

        # Handle missing values
        print("Checking for missing values...")
        if pets_data.isnull().sum().any():
            print("Missing values found. Handling missing values...")
            pets_data = handle_missing_values(pets_data)

        # Handle outliers
        print("Checking for outliers...")
        pets_data = handle_outliers(pets_data, float_cols)

        # One-hot encoding
        print("Data Processing Stage2: One Hot Encoding")
        pets_data = one_hot_encode(pets_data, cat_cols)

        # Clean column names
        print("Cleaning column names")
        pets_data = clean_column_names(pets_data)

        # Feature scaling
        print("Data Processing Stage3: MinMax Scaling")
        pets_data = scale_features(pets_data, float_cols)

        # Save processed data
        print(f"Data Processing completed. Saving file to {output_file}")
        save_data(pets_data, output_file)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
