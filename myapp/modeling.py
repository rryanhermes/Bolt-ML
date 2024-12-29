from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np
import pandas as pd
from collections import Counter

def transform(data, target_column=None):
    """Transform the input data by encoding categorical variables."""
    df = data.copy()
    
    # Identify column types
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Remove target column from categorical columns if it exists
    if target_column and target_column in categorical_columns:
        categorical_columns = categorical_columns.drop(target_column)
    
    # Handle categorical columns with one-hot encoding
    if len(categorical_columns) > 0:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Fill any missing values in numerical columns with median
    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def calculate_classification_metric(y_true, y_pred, y_pred_proba, metric_name):
    """Calculate the specified classification metric."""
    metric_name = metric_name.lower()
    if metric_name == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric_name == 'precision':
        return precision_score(y_true, y_pred, average='weighted')
    elif metric_name == 'recall':
        return recall_score(y_true, y_pred, average='weighted')
    elif metric_name == 'f1':
        return f1_score(y_true, y_pred, average='weighted')
    elif metric_name == 'auc':
        # For multiclass, calculate macro average of one-vs-rest AUC
        return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    elif metric_name == 'log loss':
        return log_loss(y_true, y_pred_proba)
    elif metric_name.startswith(('p@k', 'ap@k', 'map@k')):
        k = int(metric_name.split('k')[1]) if '@k' in metric_name else 5
        if metric_name.startswith('p@k'):
            # Precision at k implementation
            return np.mean([1 if true in pred[:k] else 0 
                          for true, pred in zip(y_true, y_pred_proba.argsort(axis=1)[:,::-1])])
        elif metric_name.startswith('ap@k'):
            # Average precision at k implementation
            precisions = []
            for true, pred in zip(y_true, y_pred_proba.argsort(axis=1)[:,::-1]):
                hits = [1 if true in pred[:i+1] else 0 for i in range(k)]
                precisions.append(np.mean(hits))
            return np.mean(precisions)
        else:  # map@k
            return np.mean([calculate_classification_metric(y_true, y_pred, y_pred_proba, f'ap@k{k}')])
    return 0.0

def calculate_regression_metric(y_true, y_pred, metric_name):
    """Calculate the specified regression metric."""
    metric_name = metric_name.lower()
    if metric_name == 'mae':
        return mean_absolute_error(y_true, y_pred)
    elif metric_name == 'mse':
        return mean_squared_error(y_true, y_pred)
    elif metric_name == 'rmse':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric_name == 'rmsle':
        return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    elif metric_name == 'mpe':
        return np.mean((y_true - y_pred) / y_true) * 100
    elif metric_name == 'mape':
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    elif metric_name == 'r2':
        return r2_score(y_true, y_pred)
    return 0.0

class ModelWrapper:
    def __init__(self, model, label_encoder=None, problem_type=None):
        self.model = model
        self.label_encoder = label_encoder
        self.problem_type = problem_type
    
    def predict(self, X):
        predictions = self.model.predict(X)
        if self.label_encoder and self.problem_type == 'classification':
            return self.label_encoder.inverse_transform(predictions)
        return predictions

def train_model(X, y, problem_type, metric_name='accuracy'):
    """Train a model based on the problem type and evaluate using specified metric."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    label_encoder = None
    if problem_type.lower() == 'classification':
        # Convert target to numeric if it's categorical
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
        
        # Train classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate the specified metric
        score = calculate_classification_metric(y_test, y_pred, y_pred_proba, metric_name)
        
        # Wrap the model with the label encoder
        wrapped_model = ModelWrapper(model, label_encoder, 'classification')
        
    elif problem_type.lower() == 'regression':
        # Train regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate the specified metric
        score = calculate_regression_metric(y_test, y_pred, metric_name)
        
        # Wrap the model without label encoder for regression
        wrapped_model = ModelWrapper(model, None, 'regression')
    
    return wrapped_model, score, metric_name 