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
    for model in models:
        print(f"Model: {model.name}, Type: {model.model_type}, Metrics: {model.metrics}")  # Debug log
    return render(request, 'my_models.html', {
        'models': models,
        'username': request.user.username
    })

@login_required
def upload_csv(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        csv_file = request.FILES['file']
        if not csv_file.name.endswith('.csv'):
            return JsonResponse({'error': 'File must be a CSV'}, status=400)
        
        # Read and process the CSV file
        df = pd.read_csv(csv_file)
        
        # Store the original dataframe in the session for later use
        request.session['original_data'] = df.to_json()
        request.session['columns'] = list(df.columns)
        
        # Transform the data
        transformed_df = transform(df)
        request.session['transformed_data'] = transformed_df.to_json()
        
        # Return column names and basic statistics
        description = df.describe().to_html()
        
        return JsonResponse({
            'columns': list(df.columns),
            'description': description,
            'plots': []  # Add any plots you want to show
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@login_required
def build_model(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        target_column = data.get('target_column')
        problem_type = data.get('problem_type')
        file_format = data.get('file_format', 'pkl')
        
        print(f"Building model for user {request.user.username}")  # Debug log
        
        # Get the data from session
        transformed_data = request.session.get('transformed_data')
        if not transformed_data:
            print("No transformed data found in session")  # Debug log
            return JsonResponse({'error': 'No data found. Please upload a CSV file first.'}, status=400)
        
        df = pd.read_json(transformed_data)
        
        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        print(f"Training model with target column: {target_column}")  # Debug log
        
        # Train the model
        model, accuracy = train_model(X, y, problem_type)
        print(f"Model trained with accuracy: {accuracy}")  # Debug log
        
        # Save the model file
        models_dir = os.path.join(settings.MEDIA_ROOT, 'models', str(request.user.id))
        os.makedirs(models_dir, exist_ok=True)
        
        model_name = f"model_{target_column}_{problem_type}"
        file_extension = '.pkl' if file_format == 'pkl' else '.joblib'
        file_path = os.path.join(models_dir, model_name + file_extension)
        
        print(f"Saving model to: {file_path}")  # Debug log
        
        # Save the model file
        if file_format == 'pkl':
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            joblib.dump(model, file_path)
        
        # Create UserModel entry
        relative_path = os.path.join('models', str(request.user.id), model_name + file_extension)
        metrics = {
            'accuracy': float(accuracy),
            'model_type': problem_type,
            'target_column': target_column
        }
        
        user_model = UserModel.objects.create(
            user=request.user,
            name=model_name,
            model_type=problem_type,
            file_path=relative_path,
            metrics=metrics
        )
        print(f"Created UserModel entry with ID: {user_model.id}")  # Debug log
        
        # Return the model file
        with open(file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="{model_name}{file_extension}"'
            response['X-Model-Accuracy'] = str(accuracy)
            return response
            
    except Exception as e:
        print(f"Error in build_model: {str(e)}")  # Debug log
        return JsonResponse({'error': str(e)}, status=400)

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
