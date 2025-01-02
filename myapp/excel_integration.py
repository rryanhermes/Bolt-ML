import xlwings as xw
import pickle
import pandas as pd
from .modeling import ModelWrapper  # Import your existing ModelWrapper class
from myapp.modeling import transform  # Import the transform function

class ExcelModelIntegration:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
    
    def load_model(self):
        """Load the saved ModelWrapper instance"""
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

if __name__ == "__main__":
    model_path = 'models/loans.pkl'
    input_data = pd.read_csv('models/Loan_Data_Test.csv')
    
    preprocessed_data = transform(input_data, target_column='Loan_Status', problem_type='classification')
    model_integration = ExcelModelIntegration(model_path)

    predictions = model_integration.predict(preprocessed_data.values)
    print(predictions)