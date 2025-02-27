import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest_classifier': RandomForestClassifier(random_state=42),
            'random_forest_regressor': RandomForestRegressor(random_state=42),
            'xgboost_classifier': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'xgboost_regressor': xgb.XGBRegressor(random_state=42)
        }
    
    def train_model(self, df, feature_columns, target_column, test_size=0.2, model_type='random_forest'):
        """Train a classification model for churn prediction"""
        # Prepare data
        X = df[feature_columns]
        y = df[target_column]
        
        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Select and train the model
        if model_type.lower() == 'xgboost':
            if len(np.unique(y)) <= 2:  # Binary classification
                model = self.models['xgboost_classifier']
            else:
                model = self.models['xgboost_regressor']
        else:  # Default to random forest
            if len(np.unique(y)) <= 2:  # Binary classification
                model = self.models['random_forest_classifier']
            else:
                model = self.models['random_forest_regressor']
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Add feature importances
        metrics['feature_importances'] = self._get_feature_importances(model, X.columns)
        
        return model, metrics
    
    def train_forecast_model(self, df, feature_columns, target_column, time_column, 
                             forecast_periods=12, test_size=0.2, model_type='xgboost'):
        """Train a time series forecasting model for sales prediction"""
        # Process time series data
        df_copy = df.copy()
        
        # Convert time column to datetime if it's not already
        if df_copy[time_column].dtype != 'datetime64[ns]':
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])
        
        # Sort by time
        df_copy = df_copy.sort_values(by=time_column)
        
        # Extract time features
        df_copy['year'] = df_copy[time_column].dt.year
        df_copy['month'] = df_copy[time_column].dt.month
        df_copy['day'] = df_copy[time_column].dt.day
        df_copy['day_of_week'] = df_copy[time_column].dt.dayofweek
        df_copy['quarter'] = df_copy[time_column].dt.quarter
        
        # Create lagged features
        for lag in range(1, 7):  # Create 6 lag features
            df_copy[f'{target_column}_lag_{lag}'] = df_copy[target_column].shift(lag)
        
        # Drop rows with NaN values from the lagged features
        df_copy = df_copy.dropna()
        
        # Prepare features and target
        X = df_copy[feature_columns + 
                   [col for col in df_copy.columns if col.startswith(f'{target_column}_lag_')] +
                   ['year', 'month', 'day', 'day_of_week', 'quarter']]
        y = df_copy[target_column]
        
        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Split the data - use the last test_size portion as the test set to mimic real forecasting
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Select and train the model
        if model_type.lower() == 'xgboost':
            model = self.models['xgboost_regressor']
        else:
            model = self.models['random_forest_regressor']
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_regression_metrics(y_test, y_pred)
        
        # Add feature importances
        metrics['feature_importances'] = self._get_feature_importances(model, X.columns)
        
        # Store the last known values for future forecasting
        self.last_known_values = df_copy.iloc[-forecast_periods:]
        
        return model, metrics
    
    def predict(self, model, data):
        """Make predictions using a trained model"""
        # Convert data to DataFrame if it's a dictionary
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Handle categorical variables
        data = pd.get_dummies(data, drop_first=True)
        
        # Ensure all columns from training are present
        for col in model.feature_names_in_:
            if col not in data.columns:
                data[col] = 0
        
        # Ensure columns are in the same order as during training
        data = data[model.feature_names_in_]
        
        # Make predictions
        predictions = model.predict(data)
        
        # If it's a classification model, get probabilities too
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data)
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist()
            }
        else:
            return {
                'predictions': predictions.tolist()
            }
    
    def forecast(self, model, forecast_periods, data=None):
        """Generate a time series forecast"""
        if not hasattr(self, 'last_known_values'):
            if data is None:
                raise ValueError("No historical data available for forecasting")
            else:
                self.last_known_values = data
        
        forecasts = []
        current_data = self.last_known_values.copy()
        
        for i in range(forecast_periods):
            # Prepare input for the next forecast
            input_data = current_data.iloc[-1:].copy()
            
            # Update time features for the next period
            last_date = pd.to_datetime(input_data.iloc[0]['date'])
            next_date = last_date + pd.DateOffset(months=1)  # Assuming monthly forecasting
            
            input_data['date'] = next_date
            input_data['year'] = next_date.year
            input_data['month'] = next_date.month
            input_data['day'] = next_date.day
            input_data['day_of_week'] = next_date.dayofweek
            input_data['quarter'] = next_date.quarter
            
            # Handle lagged features
            target_col = [col for col in input_data.columns if col.endswith('_lag_1')][0].replace('_lag_1', '')
            
            for lag in range(6, 0, -1):
                if lag > 1:
                    lag_col = f'{target_col}_lag_{lag}'
                    prev_lag_col = f'{target_col}_lag_{lag-1}'
                    input_data[lag_col] = current_data.iloc[-1][prev_lag_col]
                else:
                    input_data[f'{target_col}_lag_1'] = current_data.iloc[-1][target_col]
            
            # Make prediction
            prediction = self.predict(model, input_data)['predictions'][0]
            
            # Add prediction to the forecast list
            forecasts.append({
                'date': next_date.strftime('%Y-%m-%d'),
                'forecast': prediction
            })
            
            # Update current data for the next iteration
            input_data[target_col] = prediction
            current_data = pd.concat([current_data, input_data])
        
        return forecasts
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate classification metrics"""
        metrics = {}
        
        if len(np.unique(y_true)) <= 2:  # Binary classification
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            
            # ROC AUC only if we have binary targets
            if len(np.unique(y_true)) == 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                except:
                    pass
        else:  # Multi-class classification
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def _get_feature_importances(self, model, feature_names):
        """Extract feature importances from the model"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            return [{
                'feature': feature,
                'importance': float(importance)  # Convert to float for JSON serialization
            } for feature, importance in feature_importance]
        else:
            return []
            
    # NEW METHODS BELOW
    
    def train_classification_model(self, df, feature_columns, target_column, problem_type='binary', model_type='random_forest', model_params=None):
        """
        Train a classification model for any classification problem (binary or multiclass)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input dataframe
        feature_columns : list
            List of column names to use as features
        target_column : str
            The target column name
        problem_type : str
            'binary' or 'multiclass'
        model_type : str
            'random_forest', 'xgboost', or 'logistic'
        model_params : dict
            Additional parameters for the model
            
        Returns:
        --------
        model : trained model
        metrics : dict
            Performance metrics
        """
        # Prepare data
        X = df[feature_columns]
        y = df[target_column]
        
        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Encode target for multiclass if needed
        if problem_type == 'multiclass':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            class_labels = self.label_encoder.classes_.tolist()
        else:
            class_labels = [0, 1]  # Default for binary
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Select and train the model
        if model_params is None:
            model_params = {}
        
        if model_type == 'xgboost':
            if problem_type == 'multiclass':
                model = xgb.XGBClassifier(objective='multi:softprob', num_class=len(np.unique(y)),
                                          random_state=42, **model_params)
            else:
                model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, **model_params)
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=42, **model_params)
        else:  # Default to random forest
            model = RandomForestClassifier(random_state=42, **model_params)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        
        # Calculate metrics
        if problem_type == 'binary':
            metrics = self._calculate_binary_metrics(y_test, y_pred, y_proba)
        else:
            metrics = self._calculate_multiclass_metrics(y_test, y_pred, y_proba)
        
        # Add feature importances
        metrics['feature_importances'] = self._get_feature_importances(model, X.columns)
        
        # Add confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        metrics['class_labels'] = class_labels
        
        # Add class distribution for multiclass
        if problem_type == 'multiclass':
            class_distribution = {}
            for i, label in enumerate(class_labels):
                class_distribution[label] = int((y == i).sum())
            metrics['class_distribution'] = class_distribution
        
        # Save the model and encoder if needed
        import pickle
        model_path = os.path.join('models', 'classification_model.pkl')
        os.makedirs('models', exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_columns': list(X.columns),
                'problem_type': problem_type,
                'label_encoder': getattr(self, 'label_encoder', None)
            }, f)
        
        return model, metrics
    
    def predict_classification(self, model_data, data, problem_type='binary'):
        """
        Make predictions using a trained classification model
        
        Parameters:
        -----------
        model_data : dict
            Dictionary containing model and metadata
        data : dict or DataFrame
            Data for prediction
        problem_type : str
            'binary' or 'multiclass'
            
        Returns:
        --------
        prediction : dict
            Prediction results
        """
        # Extract model and metadata
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        label_encoder = model_data.get('label_encoder')
        
        # Convert data to DataFrame if it's a dictionary
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Handle categorical variables
        data = pd.get_dummies(data, drop_first=True)
        
        # Ensure all columns from training are present
        for col in feature_columns:
            if col not in data.columns:
                data[col] = 0
        
        # Ensure we only use columns that were in the training data
        missing_cols = set(feature_columns) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
        data = data[feature_columns]
        
        # Make predictions
        predictions = model.predict(data)
        
        result = {
            'predictions': predictions.tolist()
        }
        
        # Add probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data)
            result['probabilities'] = probabilities.tolist()
        
        # Convert predicted labels back to original classes for multiclass
        if problem_type == 'multiclass' and label_encoder is not None:
            result['predictions'] = label_encoder.inverse_transform(predictions).tolist()
        
        return result
    
    def _calculate_binary_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate metrics for binary classification"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Add ROC AUC if probabilities are available
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        
        return metrics
    
    def _calculate_multiclass_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate metrics for multiclass classification"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add class-specific metrics
        n_classes = len(np.unique(y_true))
        metrics['class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics['class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics['class_f1'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        
        return metrics
