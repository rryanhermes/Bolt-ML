from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from myapp.forms import SignUpForm
from django.contrib.auth import authenticate, login as auth_login
import pandas as pd
import plotly.express as px
import plotly.io as pio
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from django.views.decorators.csrf import csrf_exempt
import json
import pickle
import io
import joblib

def upload_csv(request):
    if request.method == 'POST' and request.FILES.get('file'):
        csv_file = request.FILES['file']
        try:
            # Add logging for file details
            print(f"Received file: {csv_file.name}, size: {csv_file.size} bytes")
            
            start_time = time.time()
            df = pd.read_csv(csv_file)
            
            # Add error checking for empty dataframes
            if df.empty:
                print("Error: Empty dataframe")
                return JsonResponse({'error': 'The uploaded CSV file is empty'}, status=400)
            
            end_time = time.time()
            loading_time = end_time - start_time

            head = df.head()
            length = df.shape[0]

            # Wrap plot generation in try-except to catch any plotting errors
            try:
                plots = generate_distribution_grid(df)
            except Exception as plot_error:
                print(f"Error generating plots: {str(plot_error)}")
                return JsonResponse({
                    'error': f'Error generating visualizations: {str(plot_error)}'
                }, status=400)

            response_data = {
                'description': head.to_html(),
                'info': f'Number of rows: {length}',
                'loading_time': f'Loading time: {loading_time:.2f} seconds',
                'plots': plots,
                'columns': df.columns.tolist()
            }

            # Store the dataframe in session
            request.session['df'] = df.to_json()
            
            return JsonResponse(response_data)

        except pd.errors.EmptyDataError:
            print("Error: Empty data error")
            return JsonResponse({'error': 'The uploaded file is empty'}, status=400)
        except pd.errors.ParserError:
            print("Error: CSV parser error")
            return JsonResponse({'error': 'Invalid CSV format'}, status=400)
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            return JsonResponse({'error': f'Error processing file: {str(e)}'}, status=400)
    else:
        print("Error: No file provided or invalid request method")
        return JsonResponse({'error': 'No file provided or invalid request method'}, status=400)

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
    
    # Update specs array - remove the invalid 'domain' parameter
    specs = [[{"type": "xy" if column_types[col] == 'numerical' else "pie"} 
             for col in df.columns]]
    
    fig = make_subplots(
        rows=1,
        cols=num_cols,
        subplot_titles=df.columns,
        specs=specs,
        horizontal_spacing=0.05
    )

    # Create visualizations in original column order
    for i, column in enumerate(df.columns):
        col = i + 1  # plotly uses 1-based indexing
        
        if column_types[column] == 'numerical':
            histogram_fig = px.histogram(df, x=column, 
                                      color_discrete_sequence=["skyblue"],
                                      opacity=0.7,
                                      nbins=30)  # Set consistent number of bins
            for trace in histogram_fig.data:
                fig.add_trace(trace, row=1, col=col)
            fig.update_xaxes(title_text=column, row=1, col=col)
            fig.update_yaxes(title_text='Count', row=1, col=col)
        
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
                        textinfo='percent',  # Show only percentages to reduce clutter
                        hoverinfo='label+value+percent',  # Show all info on hover
                        insidetextorientation='horizontal',  # Make text easier to read
                        rotation=90  # Start from top
                    ), row=1, col=col)
                except ValueError as e:
                    fig.add_annotation(
                        xref='paper', yref='paper',
                        x=(col - 0.5) / num_cols,
                        y=0.5,
                        text=f'Too many unique values<br>Count: {len(category_counts)}',
                        showarrow=False,
                        font=dict(size=10)
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

    # Update layout with improved aesthetics
    fig.update_layout(
        title_text='',
        title_x=0.5,
        height=500,  # Increased height for better visibility
        width=max(350 * num_cols, 1200),  # Increased width per column
        showlegend=False,
        margin=dict(t=100, b=50, l=50, r=50),  # Better margins
        plot_bgcolor='white',  # White background
        paper_bgcolor='white'
    )
    
    # Update all subplot titles to be more prominent
    fig.update_annotations(font_size=12)
    
    # Add grid lines to numerical plots
    for i, column in enumerate(df.columns):
        if column_types[column] == 'numerical':
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=i+1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=i+1)

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

@csrf_exempt
def build_model(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            target_column = data['target_column']
            problem_type = data['problem_type']
            file_format = data['file_format']
            
            # Get the stored dataframe
            df = pd.read_json(request.session['df'])
            
            # Import the modeling functions
            from myapp.modeling import transform, train_model
            
            # Transform the data
            X = transform(df.drop(target_column, axis=1))
            y = df[target_column]
            
            # Train the model
            model, accuracy = train_model(X, y, problem_type)
            
            # Save model to bytes buffer
            buffer = io.BytesIO()
            
            # Use appropriate format and extension
            if file_format == 'joblib':
                joblib.dump(model, buffer)
                extension = '.joblib'
            else:  # pkl
                pickle.dump(model, buffer)
                extension = '.pkl'
                
            buffer.seek(0)
            
            # Create filename with correct extension
            filename = f"model_{target_column}_{problem_type}{extension}"
            
            # Create response with file
            response = HttpResponse(buffer.getvalue(), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            response['X-Model-Accuracy'] = str(round(float(accuracy), 4))
            
            return response
            
        except Exception as e:
            print(f"Error in build_model: {str(e)}")
            return JsonResponse({
                'error': str(e)
            }, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)