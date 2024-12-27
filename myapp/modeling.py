from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
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

def train_model(X, y, problem_type):
    """Train a model based on the problem type."""
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
        
        # Get accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Wrap the model with the label encoder
        wrapped_model = ModelWrapper(model, label_encoder, 'classification')
        
    elif problem_type.lower() == 'regression':
        # Train regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get R² score as accuracy for regression
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        
        # Wrap the model without label encoder for regression
        wrapped_model = ModelWrapper(model, None, 'regression')
    
    return wrapped_model, accuracy 