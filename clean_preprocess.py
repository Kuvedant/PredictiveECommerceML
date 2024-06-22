import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split

def load_data(url):
    data = pd.read_csv(url)
    return data

def preprocess_data(data):
    data.drop_duplicates(inplace=True)
    data['Month'] = data['Month'].astype('category')
    data['Weekend'] = data['Weekend'].astype('bool')
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    transformed_df = data.copy()
    for col in data.select_dtypes(include=['float', 'int']).columns:
        if data[col].skew() > 0.5:
            data_shifted = data[col] - data[col].min() + 1
            transformed_column_name = f'{col}_transformed'
            transformed_df[transformed_column_name], _ = boxcox(data_shifted)
            transformed_df.drop(columns=[col], inplace=True)
    if set(['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']).issubset(transformed_df.columns):
        transformed_df['SessionLength'] = transformed_df[['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']].sum(axis=1)
    if set(['Administrative', 'Informational', 'ProductRelated']).issubset(transformed_df.columns):
        transformed_df['TotalActions'] = transformed_df[['Administrative', 'Informational', 'ProductRelated']].sum(axis=1)
    if set(['BounceRates', 'ExitRates']).issubset(transformed_df.columns):
        transformed_df['CombinedBounceRate'] = transformed_df['BounceRates'] + transformed_df['ExitRates']
    if 'Weekend' in transformed_df.columns:
        transformed_df['IsWeekend'] = transformed_df['Weekend'].astype(int)
    categorical_cols = ['Month', 'VisitorType']
    data_encoded = pd.get_dummies(transformed_df, columns=categorical_cols, drop_first=True)
    return data_encoded

def split_and_scale_data(data_encoded):
    X = data_encoded.drop('Revenue', axis=1)
    y = data_encoded['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
