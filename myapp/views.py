# filepath: /Users/ryanhermes/Desktop/BIZTECH/myproject/myapp/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from myapp.forms import SignUpForm
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from .models import UserModel
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from django.conf import settings
import joblib
import pickle
import pandas as pd
from .modeling import transform, train_model
from datetime import datetime
import numpy as np
import tempfile
import shutil
import zipfile
from django.utils import timezone
import pytz
import plotly.express as px
import plotly.graph_objects as go

def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            auth_login(request, user)  # Log the user in
            return redirect('index')  # Redirect to the index or another page
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, 'login.html')

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])  # Hash the password
            user.save()
            messages.success(request, "Registration successful! You can now log in.")
            return redirect('login')  # Redirect to the login page
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})

def index(request):
    context = {'username': request.user.username} if request.user.is_authenticated else {}
    return render(request, 'index.html', context)

def my_models(request):
    if not request.user.is_authenticated:
        return render(request, 'my_models.html', {'not_authenticated': True})
    
    models = UserModel.objects.filter(user=request.user)
    print(f"Found {models.count()} models for user {request.user.username}")  # Debug log
    
    # Add file size information to each model
    for model in models:
        file_path = os.path.join(settings.MEDIA_ROOT, model.file_path)
        try:
            size_bytes = os.path.getsize(file_path)
            # Convert to appropriate unit
            if size_bytes < 1024:
                model.file_size = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                model.file_size = f"{size_bytes/1024:.1f} KB"
            else:
                model.file_size = f"{size_bytes/(1024*1024):.1f} MB"
        except (OSError, FileNotFoundError):
            model.file_size = "Unknown"
    
    return render(request, 'my_models.html', {
        'models': models,
        'username': request.user.username
    })

def validate_csv(df):
    """
    Validates if a CSV file is suitable for model building.
    Returns (is_valid, message) tuple.
    """
    try:
        # Check if dataframe is empty
        if df.empty:
            return False, "The CSV file is empty"
            
        # Check if there are any columns
        if len(df.columns) < 2:
            return False, "CSV must have at least 2 columns (features and target)"
            
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            return False, "CSV contains duplicate column names"
            
        # Check for unnamed columns
        if any(not col or str(col).startswith('Unnamed: ') for col in df.columns):
            return False, "CSV contains unnamed columns. All columns must have headers"
            
        # Check for empty columns
        empty_cols = [col for col in df.columns if df[col].isna().all()]
        if empty_cols:
            return False, f"The following columns are empty: {', '.join(empty_cols)}"
            
        # Check if there's enough data
        if len(df) < 10:
            return False, "CSV must contain at least 10 rows of data for model building"
            
        # Check if there's at least one numeric or categorical column
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
        if len(numeric_cols) + len(categorical_cols) == 0:
            return False, "CSV must contain at least one numeric or categorical column"
            
        # All checks passed
        return True, "CSV is valid for model building"
        
    except Exception as e:
        return False, f"Error validating CSV: {str(e)}"

def prepare_dataset_insights(df):
    """Helper function to prepare dataset insights with proper handling of numeric/non-numeric columns"""
    # Get numeric columns for correlation and description
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Calculate correlation only for numeric columns
    correlation_html = ""
    if len(numeric_cols) > 0:
        correlation = df[numeric_cols].corr().round(2)
        
        # Create correlation heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            colorscale='RdBu',  # Red-Blue diverging colorscale
            zmid=0,  # Center the colorscale at 0
            text=np.round(correlation.values, 2),  # Show correlation values as text
            texttemplate='%{text}',
            textfont={"size": 12, "color": "black"},  # Make text larger and white
            hoverongaps=False,
            hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>'
        ))
        
        # Update layout for dark theme
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#2d2d2d',
            margin=dict(t=30, l=0, r=0, b=0),
            height=600,
            font=dict(color='white', size=12),  # Make all text white and larger
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=12, color='white')  # Make axis labels white
            ),
            yaxis=dict(
                tickfont=dict(size=12, color='white')  # Make axis labels white
            )
        )
        
        correlation_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Calculate description only for numeric columns
    description_html = ""
    if len(numeric_cols) > 0:
        description = df[numeric_cols].describe().round(2)
        description_html = description.to_html(classes='table table-dark table-hover')
    
    # Create summary table
    rows, _ = df.shape
    summary_data = []
    
    for column in df.columns:
        null_count = int(df[column].isna().sum())
        null_percentage = round((null_count / rows) * 100, 2) if null_count > 0 else 0
        unique_count = df[column].nunique()
        unique_percentage = round((unique_count / rows) * 100, 2)
        dtype = df[column].dtype
        
        summary_data.append({
            'Column': column,
            'Type': str(dtype),
            'Unique Values': f"{unique_count} ({unique_percentage}%)",
            'Null Count': f"{null_count} ({null_percentage}%)",
            'First Value': str(df[column].iloc[0]) if not pd.isna(df[column].iloc[0]) else "NULL",
            'Last Value': str(df[column].iloc[-1]) if not pd.isna(df[column].iloc[-1]) else "NULL"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_html = summary_df.to_html(classes='table table-dark table-hover', index=False)
    
    return {
        'header': df.head().to_html(classes='table table-dark table-hover'),
        'description': description_html,
        'correlation': correlation_html,
        'summary': {
            'table': summary_html,
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        }
    }

@login_required
def upload_csv(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        csv_file = request.FILES['file']
        if not csv_file.name.endswith('.csv'):
            return JsonResponse({
                'error': 'File must be a CSV',
                'is_valid': False,
                'message': 'Invalid file type. Please upload a CSV file.'
            }, status=400)
        
        # Store original filename in session (without path)
        request.session['original_filename'] = os.path.basename(csv_file.name)
        
        try:
            # Read and process the CSV file
            df = pd.read_csv(csv_file)
            
            # Validate the CSV
            is_valid, message = validate_csv(df)
            
            if not is_valid:
                return JsonResponse({
                    'error': message,
                    'is_valid': False,
                    'message': message
                }, status=400)
            
            # Store the original dataframe in the session
            request.session['original_data'] = df.to_json(orient='split', date_format='iso')
            request.session['columns'] = list(df.columns)
            
            # Transform the data without encoding the target column yet
            transformed_df = transform(df)  # No target column specified during upload
            request.session['transformed_data'] = transformed_df.to_json(orient='split', date_format='iso')
            
            # Force session save
            request.session.modified = True
            
            # Prepare dataset insights
            dataset_insights = prepare_dataset_insights(df)
            
            return JsonResponse({
                'columns': list(df.columns),
                'dataset_insights': dataset_insights,
                'is_valid': True,
                'message': message
            })
            
        except pd.errors.EmptyDataError:
            return JsonResponse({
                'error': 'The uploaded CSV file is empty',
                'is_valid': False,
                'message': 'The CSV file is empty'
            }, status=400)
        except pd.errors.ParserError:
            return JsonResponse({
                'error': 'Invalid CSV format. Please check your file.',
                'is_valid': False,
                'message': 'The file is not a valid CSV. Please check the format.'
            }, status=400)
            
    except Exception as e:
        print(f"Error in upload_csv: {str(e)}")  # Add debug logging
        return JsonResponse({
            'error': str(e),
            'is_valid': False,
            'message': 'An error occurred while processing the file.'
        }, status=400)

def format_metric_name(metric_name):
    """Format metric name for display"""
    metric_map = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score',
        'auc': 'AUC',
        'r2': 'R² Score',
        'mse': 'Mean Squared Error',
        'rmse': 'Root Mean Squared Error',
        'mae': 'Mean Absolute Error'
    }
    return metric_map.get(metric_name, metric_name.upper())

@login_required
def build_model(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        target_column = data.get('target_column')
        problem_type = data.get('problem_type')
        evaluation_metric = data.get('evaluation_metric', 'accuracy')  # Default to accuracy
        
        if not target_column or not problem_type:
            return JsonResponse({'error': 'Missing required fields: target_column and problem_type'}, status=400)
        
        # Get the data from the session
        original_data = request.session.get('original_data')
        transformed_data = request.session.get('transformed_data')
        original_filename = request.session.get('original_filename', 'model')
        
        if not original_data or not transformed_data:
            return JsonResponse({'error': 'No data found. Please upload a CSV file first.'}, status=400)
        
        try:
            # Parse the JSON data back into DataFrames
            df = pd.read_json(original_data, orient='split')
            transformed_df = pd.read_json(transformed_data, orient='split')
            
            # Verify the data is valid
            if df.empty or transformed_df.empty:
                return JsonResponse({'error': 'Invalid data in session. Please upload the CSV file again.'}, status=400)
            
            # Verify target column exists
            if target_column not in df.columns:
                return JsonResponse({'error': f'Target column "{target_column}" not found in dataset'}, status=400)
            
        except Exception as e:
            print(f"Error parsing session data: {str(e)}")
            return JsonResponse({'error': 'Invalid data format in session. Please upload the CSV file again.'}, status=400)
        
        # Prepare features and target
        try:
            X = transformed_df.drop(columns=[target_column])
            # Ensure y is 1D array
            y = df[target_column]
            if len(y.shape) > 1:
                # If y is 2D, take the first column
                y = y.iloc[:, 0] if y.shape[1] > 0 else y
            # Convert to numpy array and ensure 1D
            y = np.array(y).ravel()
        except KeyError:
            return JsonResponse({'error': f'Target column {target_column} not found in dataset'}, status=400)
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return JsonResponse({'error': 'Error preparing data for training'}, status=400)
        
        # Train the model
        try:
            model_wrapper, score, metric_name, additional_metrics = train_model(X, y, problem_type, evaluation_metric)
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return JsonResponse({'error': f'Error training model: {str(e)}'}, status=500)
        
        # Save the model
        try:
            # Get file format from request or default to pkl
            file_format = data.get('file_format', 'pkl')
            if file_format not in ['pkl', 'joblib']:
                file_format = 'pkl'  # Default to pkl if invalid format specified
            
            # Remove .csv extension if present and clean filename
            base_filename = original_filename.lower().replace('.csv', '').replace(' ', '_')
            # Create model name with format: filename_problemtype_metric
            model_name = f"{base_filename}_{problem_type}_{metric_name}"
            
            # Create user-specific directory under MEDIA_ROOT/models
            user_models_dir = os.path.join(settings.MEDIA_ROOT, 'models', str(request.user.id))
            os.makedirs(user_models_dir, exist_ok=True)
            
            # Set file extension based on format
            file_extension = '.pkl' if file_format == 'pkl' else '.joblib'
            model_filename = f"{model_name}{file_extension}"
            
            # Create absolute and relative paths
            abs_model_path = os.path.join(user_models_dir, model_filename)
            rel_model_path = os.path.join('models', str(request.user.id), model_filename)
            
            # Save the model in the specified format
            if file_format == 'pkl':
                with open(abs_model_path, 'wb') as f:
                    pickle.dump(model_wrapper, f)
            else:
                joblib.dump(model_wrapper, abs_model_path)
            
            # Create model record in database
            try:
                user_model = UserModel.objects.create(
                    user=request.user,
                    name=model_name,
                    model_type=problem_type,
                    file_path=rel_model_path,  # Store relative path in database
                    metrics={
                        'score': float(score),
                        'metric': metric_name,  # Use original metric name
                        'format': file_format,
                        **additional_metrics  # Include additional metrics
                    }
                )
            except Exception as e:
                print(f"Error creating model record: {str(e)}")
                # Clean up the saved model file if database entry fails
                if os.path.exists(abs_model_path):
                    os.remove(abs_model_path)
                return JsonResponse({'error': 'Error saving model information'}, status=500)
            
            # Return success response with model information
            return JsonResponse({
                'success': True,
                'model_id': user_model.id,
                'score': float(score),
                'metric': format_metric_name(metric_name),  # Format metric name for display
                'model_url': f'/download-model/{user_model.id}/',
                'filename': model_filename,
                'metrics': additional_metrics
            })
                
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return JsonResponse({'error': 'Error saving model file'}, status=500)
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        print(f"Unexpected error in build_model: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@login_required(login_url='/login/')
def my_account(request):
    total_models = UserModel.objects.filter(user=request.user).count()
    return render(request, 'my_account.html', {
        'total_models': total_models
    })

def logout(request):
    if request.method == 'POST':
        auth_logout(request)
        return redirect('login')
    return redirect('index')

@login_required
def delete_model(request, model_id):
    try:
        model = UserModel.objects.get(id=model_id, user=request.user)
        
        # Delete the actual model file
        file_path = os.path.join(settings.MEDIA_ROOT, model.file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete the database entry
        model.delete()
        return JsonResponse({'success': True})
    except UserModel.DoesNotExist:
        return JsonResponse({'error': 'Model not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@login_required
def rename_model(request, model_id):
    try:
        data = json.loads(request.body)
        new_name = data.get('new_name')
        
        if not new_name:
            return JsonResponse({'error': 'New name is required'}, status=400)
        
        model = UserModel.objects.get(id=model_id, user=request.user)
        model.name = new_name
        model.save()
        
        return JsonResponse({'success': True, 'new_name': new_name})
    except UserModel.DoesNotExist:
        return JsonResponse({'error': 'Model not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

def home(request):
    context = {'username': request.user.username} if request.user.is_authenticated else {}
    return render(request, 'home.html', context)

def build_model_page(request):
    context = {'username': request.user.username} if request.user.is_authenticated else {}
    return render(request, 'build_model.html', context)

def get_user_timezone(request):
    # Try to get timezone from session
    user_timezone = request.session.get('timezone', None)
    if not user_timezone:
        # Default to UTC if no timezone is set
        user_timezone = 'UTC'
    return pytz.timezone(user_timezone)

@login_required
def set_timezone(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_timezone = data.get('timezone', 'UTC')
            
            # Validate the timezone
            try:
                pytz.timezone(user_timezone)
                request.session['timezone'] = user_timezone
                return JsonResponse({'success': True})
            except pytz.exceptions.UnknownTimeZoneError:
                return JsonResponse({'error': 'Invalid timezone'}, status=400)
                
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

@login_required
def download_model(request, model_id):
    try:
        # Get the model object and verify ownership
        model = UserModel.objects.get(id=model_id, user=request.user)
        
        # Get user's timezone
        user_tz = get_user_timezone(request)
        
        # Convert created_at to user's timezone
        local_created_at = timezone.localtime(model.created_at, user_tz)
        
        # Get the absolute file path
        model_file_path = os.path.join(settings.MEDIA_ROOT, model.file_path)
        
        if not os.path.exists(model_file_path):
            return JsonResponse({'error': 'Model file not found'}, status=404)
        
        # Create a temporary directory for the zip contents
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory inside temp_dir for the model files
            model_dir = os.path.join(temp_dir, model.name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Copy the model file to the temp directory
            model_filename = os.path.basename(model_file_path)
            shutil.copy2(model_file_path, os.path.join(model_dir, model_filename))
            
            # Create configuration file
            config = {
                'model_name': model.name,
                'model_type': model.model_type,
                'created_at': local_created_at.isoformat(),
                'timezone': str(user_tz),
                'metrics': model.metrics,
                'file_format': model_file_path.split('.')[-1],
                'python_version': '3.8+',
                'required_packages': [
                    'scikit-learn',
                    'pandas',
                    'numpy',
                    'joblib' if model_file_path.endswith('.joblib') else 'pickle'
                ]
            }
            
            # Write configuration to JSON file
            config_path = os.path.join(model_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Create README with usage instructions
            readme_content = f"""# {model.name}

## Model Information
- Type: {model.model_type}
- Created: {local_created_at.strftime('%Y-%m-%d %H:%M:%S %Z')}
- Timezone: {user_tz}
- Metrics: {json.dumps(model.metrics, indent=2)}

## Requirements
- Python 3.8+
- Required packages: scikit-learn, pandas, numpy, {'joblib' if model_file_path.endswith('.joblib') else 'pickle'}

## Usage Instructions

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Load the model:
   ```python
   import {'joblib' if model_file_path.endswith('.joblib') else 'pickle'}
   import pandas as pd
   
   # Load the model
   model = {'joblib.load' if model_file_path.endswith('.joblib') else 'pickle.load(open'}('{model_filename}'{', "rb")' if model_file_path.endswith('.pkl') else ')'})
   
   # Prepare your data (must have the same columns as training data)
   data = pd.read_csv('your_data.csv')
   
   # Make predictions
   predictions = model.predict(data)
   ```

3. Required Data Format:
   - Input data must be a pandas DataFrame
   - Required columns: [list of required columns]
   - All numeric columns should be filled (no NaN values)

For more information, see the config.json file.
"""
            
            # Write README
            readme_path = os.path.join(model_dir, 'README.md')
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            # Create requirements.txt
            requirements_content = """scikit-learn>=0.24.0
pandas>=1.2.0
numpy>=1.19.0
joblib>=0.17.0"""
            
            requirements_path = os.path.join(model_dir, 'requirements.txt')
            with open(requirements_path, 'w') as f:
                f.write(requirements_content)
            
            # Create the zip file
            zip_filename = f"{model.name}_package.zip"
            zip_path = os.path.join(temp_dir, zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from the model directory
                for root, _, files in os.walk(model_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
            
            # Read the zip file and serve it
            with open(zip_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type='application/zip')
                response['Content-Disposition'] = f'attachment; filename="{zip_filename}"'
                return response
            
    except UserModel.DoesNotExist:
        return JsonResponse({'error': 'Model not found'}, status=404)
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

def premium(request):
    """
    View for the premium subscription page.
    """
    return render(request, 'premium.html', {
        'username': request.user.username if request.user.is_authenticated else None
    })

def blog(request):
    """
    View for the blog page.
    """
    return render(request, 'blog.html', {
        'username': request.user.username if request.user.is_authenticated else None
    })

@csrf_exempt
def newsletter_signup(request):
    """
    Handle newsletter signup submissions.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        email = data.get('email')
        
        if not email:
            return JsonResponse({'error': 'Email is required'}, status=400)
        
        # Here you would typically:
        # 1. Validate the email
        # 2. Add it to your newsletter service/database
        # 3. Send a confirmation email
        # For now, we'll just return success
        
        return JsonResponse({
            'success': True,
            'message': 'Thank you for subscribing to our newsletter!'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def about(request):
    return render(request, 'about.html')

@login_required
def get_sample_dataset(request, dataset_name):
    """Handle sample dataset requests"""
    try:
        # Map dataset names to file paths
        dataset_paths = {
            'titanic': 'models/titanic.csv',
            'housing': 'models/housingprices.csv',
            'credit': 'models/credit.csv'
        }
        
        if dataset_name not in dataset_paths:
            return JsonResponse({
                'success': False,
                'message': 'Invalid dataset name'
            }, status=400)
            
        # Get the absolute path to the dataset
        dataset_path = os.path.join(settings.BASE_DIR, dataset_paths[dataset_name])
        
        # Read the CSV file
        df = pd.read_csv(dataset_path)
        
        # Store the original dataframe in the session
        request.session['original_data'] = df.to_json(orient='split', date_format='iso')
        request.session['columns'] = list(df.columns)
        request.session['original_filename'] = f"{dataset_name}.csv"
        
        # Transform the data without encoding the target column yet
        transformed_df = transform(df)
        request.session['transformed_data'] = transformed_df.to_json(orient='split', date_format='iso')
        
        # Force session save
        request.session.modified = True
        
        # Prepare dataset insights
        dataset_insights = prepare_dataset_insights(df)
        
        return JsonResponse({
            'success': True,
            'columns': list(df.columns),
            'dataset_insights': dataset_insights,
            'message': 'Dataset loaded successfully'
        })
        
    except FileNotFoundError:
        return JsonResponse({
            'success': False,
            'message': 'Sample dataset file not found'
        }, status=404)
    except Exception as e:
        print(f"Error loading sample dataset: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Error loading dataset: {str(e)}'
        }, status=500)
