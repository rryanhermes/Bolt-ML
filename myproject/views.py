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
    # Initialize column types while maintaining order
    column_types = {}
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals == {0, 1} or unique_vals == {0.0, 1.0}:
                column_types[col] = 'binary'
            else:
                column_types[col] = 'numerical'
        else:
            column_types[col] = 'categorical'
    
    num_cols = len(df.columns)
    
    # Create specs array - one row, many columns
    specs = [[{"type": "xy" if column_types[col] == 'numerical' else "domain"} 
             for col in df.columns]]
    
    fig = make_subplots(
        rows=1,
        cols=num_cols,
        subplot_titles=df.columns,
        specs=specs
    )

    # Create visualizations in original column order
    for i, column in enumerate(df.columns):
        col = i + 1  # plotly uses 1-based indexing
        
        if column_types[column] == 'numerical':
            # Histogram for numerical columns
            histogram_fig = px.histogram(df, x=column, 
                                      color_discrete_sequence=["skyblue"],  
                                      opacity=0.7)
            for trace in histogram_fig.data:
                fig.add_trace(trace, row=1, col=col)
            fig.update_yaxes(title_text='Frequency', row=1, col=col)
        
        else:  # binary or categorical
            category_counts = df[column].value_counts()
            
            if column_types[column] == 'binary':
                # For binary columns, create custom labels
                labels = ['False (0)', 'True (1)'] if len(category_counts) == 2 else \
                        ['True (1)' if 1 in category_counts.index else 'False (0)']
                values = [category_counts.get(0, 0), category_counts.get(1, 0)] if len(category_counts) == 2 else \
                        [category_counts.iloc[0]]
            else:
                # For regular categorical columns
                labels = category_counts.index
                values = category_counts.values

            if len(category_counts) <= 20:
                try:
                    fig.add_trace(go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.3,
                        textinfo='label+percent',
                        insidetextorientation='radial'
                    ), row=1, col=col)
                except ValueError as e:
                    fig.add_annotation(
                        xref='paper', yref='paper',
                        x=(col - 0.5) / num_cols,
                        y=0.5,
                        text=f'{column}\nUnique Values: {len(category_counts)}',
                        showarrow=False,
                        font=dict(size=12)
                    )
            else:
                fig.add_annotation(
                    xref='paper', yref='paper',
                    x=(col - 0.5) / num_cols,
                    y=0.5,
                    text=f'{column}\nUnique Values: {len(category_counts)}',
                    showarrow=False,
                    font=dict(size=12)
                )

    # Update layout
    fig.update_layout(
        title_text='',
        title_x=0.5,
        height=400,  # Fixed height since we only have one row
        width=max(300 * num_cols, 1200),  # Scale width based on number of columns, minimum 1200px
        showlegend=False,
        margin=dict(t=100)  # Add more top margin for titles
    )
    
    # Convert to HTML and return as a single element list to maintain compatibility
    return [pio.to_html(fig, full_html=False)]

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