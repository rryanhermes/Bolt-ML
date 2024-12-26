from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pandas as pd
from collections import Counter

def transform(data):
    """Transform the input data by encoding categorical variables."""
    df = data.copy()
    
    # Identify column types
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Handle categorical columns
    if len(categorical_columns) > 0:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Fill any missing values in numerical columns with median
    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def train_model(X, y, problem_type):
    """Train a model based on the problem type."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if problem_type.lower() == 'classification':
        # Convert target to numeric if it's categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
        
        # Train classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
    elif problem_type.lower() == 'regression':
        # Train regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get R² score as accuracy for regression
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
    
    return model, accuracy 