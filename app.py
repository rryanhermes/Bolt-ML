from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/upload-csv/', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'is_valid': False, 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'is_valid': False, 'message': 'Invalid file type'}), 400
    
    df = pd.read_csv(file)
    request.session['df'] = df.to_json(orient='split')
    
    return jsonify({'is_valid': True, 'columns': df.columns.tolist()})

@app.route('/build-model-api/', methods=['POST'])
def excel_build_model():
    data = request.json
    target_column = data['target_column']
    problem_type = data['problem_type']
    
    df = pd.read_json(request.session['df'], orient='split')
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    model = RandomForestClassifier()
    model.fit(X, y)
    
    model_filename = f'models/{target_column}_{problem_type}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    return jsonify({'success': True, 'model_path': model_filename})

if __name__ == '__main__':
    app.run(debug=True)
