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
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def upload_csv(request):
    if request.method == 'POST' and request.FILES['file']:
        csv_file = request.FILES['file']
        try:
            start_time = time.time()  # Start time
            df = pd.read_csv(csv_file)  # Load the CSV file into a DataFrame
            end_time = time.time()  # End time
            loading_time = end_time - start_time  # Calculate loading time

            head = df.head()  # Get description
            length = df.shape[0]  # Get number of rows

            print(f'Number of rows: {length}')
            print(f'Loading time: {loading_time:.2f} seconds')

            # Generate the distribution grid
            plots = generate_distribution_grid(df)

            return JsonResponse({
                'description': head.to_html(),
                'info': f'Number of rows: {length}',
                'loading_time': f'Loading time: {loading_time:.2f} seconds',
                'plots': plots
            }, safe=False)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def generate_distribution_grid(df):
    numerical_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(exclude=['number']).columns

    plots = []
    
    # Prepare subplot layout
    num_plots = len(numerical_columns) + len(categorical_columns)
    rows = (num_plots // num_plots) + (num_plots % num_plots > 0)
    fig = make_subplots(rows=rows, cols=num_plots, subplot_titles=[])

    # Generate histograms for numerical columns
    for i, column in enumerate(numerical_columns):
        row = i // num_plots + 1
        col = i % num_plots + 1
        histogram_fig = px.histogram(df, x=column, title=f'Histogram of {column}', width=300, height=300)
        for trace in histogram_fig.data:
            fig.add_trace(trace, row=row, col=col)

    # Generate pie/bar charts for categorical columns
    for i, column in enumerate(categorical_columns):
        category_counts = df[column].value_counts()
        row = (len(numerical_columns) + i) // num_plots + 1
        col = (len(numerical_columns) + i) % num_plots + 1
        
        if len(category_counts) <= 20:
            pie_fig = px.pie(df, names=category_counts.index, values=category_counts.values, title=f'Pie Chart of {column}')
            for trace in pie_fig.data:
                fig.add_trace(trace, row=row, col=col)
        else:
            bar_fig = px.bar(x=category_counts.index, y=category_counts.values, title=f'Bar Chart of {column}')
            for trace in bar_fig.data:
                fig.add_trace(trace, row=row, col=col)

    # Update layout for better spacing
    fig.update_layout(height=300 * rows, width=300 * num_plots)
    
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