from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, 
    mean_squared_error, 
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
from collections import Counter

def transform(data, target_column=None, problem_type='classification'):
    """Transform the input data with enhanced preprocessing for better model accuracy."""
    df = data.copy()
    
    # Identify column types
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Remove target column from feature columns
    if target_column:
        if target_column in categorical_columns:
            categorical_columns = categorical_columns.drop(target_column)
        if target_column in numerical_columns:
            numerical_columns = numerical_columns.drop(target_column)
    
    # Handle numerical features
    if len(numerical_columns) > 0:
        # Handle outliers using Robust Scaler for numerical columns
        robust_scaler = RobustScaler()
        df[numerical_columns] = robust_scaler.fit_transform(df[numerical_columns])
        
        # Fill missing values with median (more robust to outliers)
        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical features
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            
            # Handle missing values first
            if null_count > 0:
                # For high cardinality, use a special "NaN" category
                if unique_count > 10:
                    df[col] = df[col].fillna("NaN")
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
            
            # Encoding strategy based on cardinality
            if unique_count <= 2:
                # Binary categories: Use label encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            elif unique_count <= 10:
                # Low cardinality: Use one-hot encoding
                df = pd.get_dummies(df, columns=[col], prefix=col)
            elif target_column and problem_type.lower() == 'classification':
                # High cardinality: Use target encoding for classification
                target_means = df.groupby(col)[target_column].mean()
                df[col] = df[col].map(target_means)
            else:
                # High cardinality: Use frequency encoding for regression or when no target
                value_counts = df[col].value_counts(normalize=True)
                df[col] = df[col].map(value_counts)
    
    # Handle target column if present
    if target_column:
        if df[target_column].dtype == 'object':
            le = LabelEncoder()
            df[target_column] = le.fit_transform(df[target_column])
    
    # Feature selection for high-dimensional data
    if df.shape[1] > 20 and target_column:  # Only if we have many features and a target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Use Random Forest for feature importance-based selection
        if problem_type.lower() == 'classification':
            selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
        else:
            selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Keep only selected features and target
        if target_column not in selected_features:
            selected_features.append(target_column)
        df = df[selected_features]
    
    return df

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
        
    def __getstate__(self):
        """Return state values to be pickled."""
        return {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'problem_type': self.problem_type
        }
    
    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.model = state['model']
        self.label_encoder = state['label_encoder']
        self.problem_type = state['problem_type']

def train_model(X, y, problem_type, evaluation_metric='accuracy'):
    """Train a model based on the problem type and evaluation metric."""
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
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate all classification metrics
        metrics = {
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Calculate primary metric based on evaluation_metric parameter
        if evaluation_metric == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif evaluation_metric == 'precision':
            score = metrics['precision']
        elif evaluation_metric == 'recall':
            score = metrics['recall']
        elif evaluation_metric == 'f1':
            score = metrics['f1']
        elif evaluation_metric == 'auc':
            from sklearn.metrics import roc_auc_score
            # For multi-class, we need to binarize the labels
            from sklearn.preprocessing import label_binarize
            classes = list(range(len(set(y_test))))
            if len(classes) > 2:
                y_test_bin = label_binarize(y_test, classes=classes)
                y_pred_proba = model.predict_proba(X_test)
                score = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
            else:
                score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        else:
            score = accuracy_score(y_test, y_pred)  # Default to accuracy
        
        # Wrap the model with the label encoder
        wrapped_model = ModelWrapper(model, label_encoder, 'classification')
        
        return wrapped_model, score, evaluation_metric, metrics
        
    elif problem_type.lower() == 'regression':
        # Train regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metric based on evaluation_metric parameter
        if evaluation_metric == 'r2':
            score = r2_score(y_test, y_pred)
        elif evaluation_metric == 'mse':
            score = mean_squared_error(y_test, y_pred)
        elif evaluation_metric == 'rmse':
            score = mean_squared_error(y_test, y_pred, squared=False)
        elif evaluation_metric == 'mae':
            from sklearn.metrics import mean_absolute_error
            score = mean_absolute_error(y_test, y_pred)
        else:
            score = r2_score(y_test, y_pred)  # Default to R²
        
        # Wrap the model without label encoder for regression
        wrapped_model = ModelWrapper(model, None, 'regression')
        
        return wrapped_model, score, evaluation_metric, None  # Return None for metrics in regression case 