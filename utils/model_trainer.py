import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {
            'xgboost_classifier': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        }
    
    def train_model(self, df, feature_columns, target_column, test_size=0.2, model_type='xgboost'):
        """Train a classification model for churn prediction"""
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
        
        # Always use XGBoost for churn prediction regardless of model_type parameter
        model = self.models['xgboost_classifier']
        
        # Set objective based on number of classes
        if self.num_classes > 2:
            model.set_params(objective='multi:softprob', num_class=self.num_classes)
        else:
            model.set_params(objective='binary:logistic')
        
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
        
        # Convert NumPy int64/float64 to Python native types for JSON serialization
        predictions_list = [int(pred) if isinstance(pred, (np.integer, np.int64)) else 
                            float(pred) if isinstance(pred, (np.floating, np.float64)) else 
                            bool(pred) if isinstance(pred, np.bool_) else 
                            str(pred) for pred in predictions]
        
        # For multiclass, convert numeric predictions to original class labels if available
        if hasattr(self, 'target_classes') and len(self.target_classes) > 0:
            # Map numeric predictions back to original class labels
            class_predictions = [str(self.target_classes[int(pred)]) for pred in predictions]
        else:
            class_predictions = predictions_list
        
        # Get probabilities for classification
        probabilities = model.predict_proba(data)
        
        # Convert probabilities to native Python types for JSON serialization
        probs_list = []
        for sample_probs in probabilities:
            probs_list.append([float(prob) for prob in sample_probs])
        
        # Convert any NumPy types in target_classes to Python native types
        target_classes_list = None
        if hasattr(self, 'target_classes'):
            target_classes_list = [str(c) if isinstance(c, np.ndarray) or isinstance(c, np.generic) 
                                else c for c in self.target_classes.tolist()]
        
        return {
            'predictions': class_predictions,
            'numeric_predictions': predictions_list,
            'probabilities': probs_list,
            'class_labels': target_classes_list
        }
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate classification metrics"""
        metrics = {}
        
        # Basic metrics for all classification problems
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        if len(np.unique(y_true)) <= 2:  # Binary classification
            metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
            
            # ROC AUC only if we have binary targets
            if len(np.unique(y_true)) == 2:
                try:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred))
                except:
                    pass
        else:  # Multi-class classification
            metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['num_classes'] = int(len(np.unique(y_true)))
        
        return metrics
    
    def _get_feature_importances(self, model, feature_names):
        """Extract feature importances from the model"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            return [{
                'feature': str(feature),
                'importance': float(importance)  # Convert to float for JSON serialization
            } for feature, importance in feature_importance]
        else:
            return []