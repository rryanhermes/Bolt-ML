import xlwings as xw
import joblib
import pandas as pd
import os
import pickle

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

class ExcelModelIntegration:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
    
    def load_model(self):
        """Load the saved ModelWrapper instance using joblib or pickle"""
        if self.model_path.endswith('.joblib'):
            return joblib.load(self.model_path)
        else:
            with open(self.model_path, 'rb') as file:
                return pickle.load(file)
    
    @xw.func
    def predict(self, input_range):
        """Excel function to make predictions"""
        try:
            # Convert Excel range to DataFrame
            df = pd.DataFrame(input_range)
            
            # Make predictions using your ModelWrapper
            predictions = self.model.predict(df)
            
            # Return predictions to Excel
            return predictions.tolist()
        except Exception as e:
            return f"Error: {str(e)}"

# Usage example - you can add this at the bottom of the file
if __name__ == "__main__":
    model_path = 'models/titanic.joblib'  # Update with your model path
    model_integration = ExcelModelIntegration(model_path)
    print(model_integration.predict([[1, 2, 3], [4, 5, 6]]))  # Example usage
