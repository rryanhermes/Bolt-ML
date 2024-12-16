from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from myapp.forms import SignUpForm
from django.contrib.auth import authenticate, login as auth_login
import pandas as pd
import plotly.express as px
import plotly.io as pio
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required

def upload_csv(request):
    if request.method == 'POST' and request.FILES['file']:
        csv_file = request.FILES['file']
        try:
            df = pd.read_csv(csv_file)  # Load the CSV file into a DataFrame
            description = df.describe()  # Get description

            # Generate the distribution grid
            plots = generate_distribution_grid(df)

            return JsonResponse({
                'description': description.to_html(),
                'plots': plots
            }, safe=False)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def generate_distribution_grid(df, num_cols=3):
    numerical_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(exclude=['number']).columns

    plots = []

    for column in numerical_columns:
        fig = px.histogram(df, x=column, title=f'Histogram of {column}')
        plots.append(pio.to_html(fig, full_html=False))

    for column in categorical_columns:
        category_counts = df[column].value_counts()
        if len(category_counts) <= 20:
            fig = px.pie(df, names=column, title=f'Pie Chart of {column}')
        else:
            fig = px.bar(df, x=category_counts.index, y=category_counts.values, title=f'Bar Chart of {column}')
        plots.append(pio.to_html(fig, full_html=False))

    return plots

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
    return render(request, 'index.html')

def testing(request):
    return render(request, 'testing.html')