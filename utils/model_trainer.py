import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest_classifier': RandomForestClassifier(random_state=42),
            'random_forest_regressor': RandomForestRegressor(random_state=42),
            'xgboost_classifier': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            'xgboost_regressor': xgb.XGBRegressor(random_state=42)
        }
    
    def train_model(self, df, feature_columns, target_column, test_size=0.2, model_type='random_forest'):
        """Train a classification model for general classification tasks"""
        # Prepare data
        X = df[feature_columns]
        y = df[target_column]
        
        # Handle categorical target if needed
        if y.dtype == 'object' or y.dtype == 'category':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            # Store the label encoder classes for prediction
            self.target_classes = label_encoder.classes_
        else:
            self.target_classes = np.unique(y)
        
        # Save the number of classes
        self.num_classes = len(np.unique(y))
        
        # Handle categorical variables in features
        X = pd.get_dummies(X, drop_first=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Select and train the model
        is_classification = True  # Default to classification
        
        if model_type.lower() == 'xgboost':
            if is_classification:
                model = self.models['xgboost_classifier']
                # Set objective based on number of classes
                if self.num_classes > 2:
                    model.set_params(objective='multi:softprob', num_class=self.num_classes)
                else:
                    model.set_params(objective='binary:logistic')
            else:
                model = self.models['xgboost_regressor']
        else:  # Default to random forest
            if is_classification:
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
        
        # Save model metadata
        self.feature_columns = feature_columns
        self.target_column = target_column
        
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
        
        # For multiclass, convert numeric predictions to original class labels if available
        if hasattr(self, 'target_classes') and len(self.target_classes) > 0:
            # Map numeric predictions back to original class labels
            class_predictions = [self.target_classes[int(pred)] for pred in predictions]
        else:
            class_predictions = predictions.tolist()
        
        # If it's a classification model, get probabilities too
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data)
            return {
                'predictions': class_predictions,
                'numeric_predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'class_labels': self.target_classes.tolist() if hasattr(self, 'target_classes') else None
            }
        else:
            return {
                'predictions': class_predictions,
            }
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate classification metrics"""
        metrics = {}
        
        # Basic metrics for all classification problems
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if len(np.unique(y_true)) <= 2:  # Binary classification
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
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['num_classes'] = len(np.unique(y_true))
        
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