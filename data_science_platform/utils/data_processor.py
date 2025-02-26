import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataProcessor:
    def __init__(self):
        self.numerical_transformer = None
        self.categorical_transformer = None
        self.preprocessor = None
    
    def load_data(self, filepath):
        """Load data from CSV or Excel file"""
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")
        
        return df
    
    def get_feature_types(self, df):
        """Identify numerical and categorical features"""
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        return numerical_features, categorical_features
    
    def create_preprocessor(self, df, numerical_features, categorical_features):
        """Create a preprocessing pipeline for numerical and categorical features"""
        # Preprocessing for numerical data
        self.numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing for categorical data
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Bundle preprocessing for numerical and categorical data
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_transformer, numerical_features),
                ('cat', self.categorical_transformer, categorical_features)
            ])
        
        return self.preprocessor
    
    def preprocess_data(self, df, numerical_features=None, categorical_features=None, fit=True):
        """Preprocess data using the defined pipeline"""
        if numerical_features is None or categorical_features is None:
            numerical_features, categorical_features = self.get_feature_types(df)
        
        if self.preprocessor is None or fit:
            self.create_preprocessor(df, numerical_features, categorical_features)
            X_processed = self.preprocessor.fit_transform(df[numerical_features + categorical_features])
        else:
            X_processed = self.preprocessor.transform(df[numerical_features + categorical_features])
        
        return X_processed
    
    def split_features_target(self, df, target_column):
        """Split dataframe into features and target"""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
    
    def handle_time_series(self, df, time_column):
        """Process time series data"""
        # Convert time column to datetime if it's not already
        if df[time_column].dtype != 'datetime64[ns]':
            df[time_column] = pd.to_datetime(df[time_column])
        
        # Sort by time
        df = df.sort_values(by=time_column)
        
        # Extract time features
        df['year'] = df[time_column].dt.year
        df['month'] = df[time_column].dt.month
        df['day'] = df[time_column].dt.day
        df['day_of_week'] = df[time_column].dt.dayofweek
        df['quarter'] = df[time_column].dt.quarter
        
        return df
    
    def create_lagged_features(self, df, target_column, lags=None):
        """Create lagged features for time series forecasting"""
        if lags is None:
            lags = [1, 2, 3, 6, 12]  # Default lags
        
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Drop rows with NaN values (from the shifts)
        df = df.dropna()
        
        return df
