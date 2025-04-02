from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, FileResponse
from Genitic_insight.forms import FastaUploadForm 
from Bio import SeqIO
from io import TextIOWrapper
import os
import time  # Added import for time module
import pandas as pd
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import sys
from pathlib import Path
# views.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from django.views.decorators.csrf import csrf_exempt
# Add your project root to Python path
project_root = Path(__file__).resolve().parent.parent  # Adjust based on your structure
sys.path.append(str(project_root))

from .proteinfeature import ProteinFeatureExtractor
from .RNAfeature import RNAFeatureExtractor
from .DNAfeature import DNAFeatureExtractor
import csv
import json
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from io import StringIO
from sklearn.model_selection import train_test_split
from django.shortcuts import render, redirect
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import joblib
from io import TextIOWrapper, StringIO
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, 
    BaggingClassifier, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, 
    QuadraticDiscriminantAnalysis
)
from sklearn.naive_bayes import GaussianNB

# Check for optional ML packages
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Create your views here.


# Home
def home(request):
    return render(request, 'home.html')

# detect_sequence_type
def detect_sequence_type(sequence):
    """
    Detect whether a biological sequence is DNA, RNA, or Protein.
    
    Args:
        sequence (str): Input biological sequence
        
    Returns:
        str: "DNA", "RNA", "Protein", or "Unknown"
    """
    # Define character sets
    protein_chars = set("ACDEFGHIKLMNPQRSTVWY*")
    dna_chars = set("ATCGN")
    rna_chars = set("AUCGN")
    ambiguous_dna = set("BDHKMNRSVWY")
    ambiguous_rna = set("BDHKMNRSVWY")
    
    # Clean and validate input
    if not sequence:
        return "Unknown"
        
    sequence = sequence.upper().strip()
    sequence = ''.join([c for c in sequence if c.isalpha() or c == '*'])
    
    if not sequence:
        return "Unknown"
    
    # Check for protein markers
    has_protein_chars = any(c in protein_chars for c in sequence)
    has_stop_codon = '*' in sequence
    
    # Check for nucleic acid markers
    has_dna = any(c in dna_chars for c in sequence)
    has_rna = any(c in rna_chars for c in sequence)
    has_u = 'U' in sequence
    has_t = 'T' in sequence
    
    # Detection logic
    if has_protein_chars and not (has_dna or has_rna):
        return "Protein"
    
    if has_u and not has_t:
        if any(c in rna_chars or c in ambiguous_rna for c in sequence):
            return "RNA"
    
    if has_t and not has_u:
        if any(c in dna_chars or c in ambiguous_dna for c in sequence):
            return "DNA"
    
    # Handle ambiguous cases
    if not has_u and not has_t:
        if has_protein_chars:
            return "Protein"
        if len(sequence) >= 3 and len(sequence) % 3 == 0:
            return "Protein" if has_protein_chars else "DNA"
    
    return "Unknown"

# feature_extraction
def feature_extraction(request):
    if request.method == 'POST' and request.FILES.get('fasta_file'):
        try:
            fasta_file = TextIOWrapper(request.FILES['fasta_file'].file, encoding='utf-8')
            record = next(SeqIO.parse(fasta_file, "fasta"))
            sequence = str(record.seq)
            
            response_data = {
                'sequence_type': detect_sequence_type(sequence),
                'sequence_id': record.id,
                'sequence_preview': sequence[:100] + ('...' if len(sequence) > 100 else ''),
                'length': len(sequence)
            }
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse(response_data)
            return render(request, 'feature_extraction.html', response_data)

        except Exception as e:
            error_response = {'error': str(e)}
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse(error_response, status=400)
            return render(request, 'feature_extraction.html', error_response)
    
    return render(request, 'feature_extraction.html', {'form': FastaUploadForm()})

def analyze_sequence(request):
    if request.method == 'POST' and request.FILES.get('fasta_file'):
        try:
            # Get uploaded file and parameters
            fasta_file = request.FILES['fasta_file']
            descriptor = request.POST.get('descriptor', '')
            sequence_type = request.POST.get('sequence_type', 'Unknown')
            parameters_str = request.POST.get('parameter', '') 
            
            # Save the file temporarily
            file_path = os.path.join(settings.MEDIA_ROOT, 'temp', fasta_file.name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb+') as destination:
                for chunk in fasta_file.chunks():
                    destination.write(chunk)
            
            # Initialize csv_data
            csv_data = None
            
            # Convert parameters string to dictionary
            params_dict = {}
            if parameters_str:
                try:
                    params_dict = dict(item.split('=') for item in parameters_str.split(','))
                    # Convert numeric values
                    for key, value in params_dict.items():
                        try:
                            params_dict[key] = float(value) if '.' in value else int(value)
                        except ValueError:
                            pass
                except Exception as e:
                    print(f"Parameter parsing error: {e}")
                    params_dict = {}
            
            # Process based on sequence type
            if sequence_type == 'Protein':
                extractor = ProteinFeatureExtractor()
                csv_data = extractor.to_csv(
                    file_path,
                    methods=[descriptor],
                    params=params_dict
                )
            elif sequence_type == 'DNA':
                extractor = DNAFeatureExtractor()
                csv_data = extractor.to_csv(
                    file_path,
                    methods=[descriptor],
                    params=params_dict
                )
            elif sequence_type == 'RNA':
                extractor = RNAFeatureExtractor()
                csv_data = extractor.to_csv(
                    file_path,
                    methods=[descriptor],
                    params=params_dict
                )
            else:
                raise ValueError(f"Unknown sequence type: {sequence_type}")
            
            # Prepare response
            response_data = {
                'message': 'Feature extraction completed',
                'file_name': fasta_file.name,
                'sequence_type': sequence_type,
                'descriptor': descriptor,
                'csv_data': csv_data
            }
            
            # Clean up temporary file
            os.remove(file_path)
            
            return JsonResponse(response_data)
            
        except Exception as e:
            # Clean up temporary file if it exists
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


# @require_POST
# def save_extracted_data(request):
#     """Save extracted features to session"""
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             csv_data = data.get('csv_data', '')
#             request.session['extracted_features'] = csv_data
#             request.session.modified = True
#             return JsonResponse({'status': 'success'})
#         except Exception as e:
#             return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
#     return JsonResponse({'status': 'error'}, status=400)

def module_selection(request):
    """Render module selection page with available algorithms"""
    return render(request, 'module_selection.html')




@csrf_exempt
def train_model(request):
    if request.method == 'POST':
        try:
            # Validate file upload
            if 'dataset_file' not in request.FILES:
                return JsonResponse({'error': 'No dataset file provided'}, status=400)
            
            # Validate parameters
            try:
                train_percent = int(request.POST.get('train_percent', 80))
                if not (0 < train_percent < 100):
                    raise ValueError
            except (ValueError, TypeError):
                return JsonResponse({'error': 'Invalid train percentage (must be integer between 1-99)'}, status=400)
            
            algorithm = request.POST.get('algorithm')
            
            # Process file
            try:
                df = pd.read_csv(request.FILES['dataset_file'])
                if len(df.columns) < 2:
                    return JsonResponse({'error': 'Dataset must have at least 2 columns (features + target)'}, status=400)
            except Exception as e:
                return JsonResponse({'error': f'Error reading CSV file: {str(e)}'}, status=400)
            
            X = df.iloc[:, :-1]  # Assume last column is target
            y = df.iloc[:, -1]
            
            # Determine problem type
            problem_type = 'classification' if y.dtype == 'object' or len(y.unique()) < 10 else 'regression'
            
            # Validate algorithm based on problem type
            regression_algorithms = ['linear_regression', 'random_forest_regressor', 'svm_regressor']
            classification_algorithms = ['logistic_regression', 'random_forest', 'decision_tree', 'svm', 'knn', 'neural_network']
            
            if problem_type == 'regression' and algorithm not in regression_algorithms:
                return JsonResponse({'error': f'Selected algorithm is not suitable for regression problems. Please use: {", ".join(regression_algorithms)}'}, status=400)
            elif problem_type == 'classification' and algorithm not in classification_algorithms:
                return JsonResponse({'error': f'Selected algorithm is not suitable for classification problems. Please use: {", ".join(classification_algorithms)}'}, status=400)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_percent/100, random_state=42
            )
            
            # Train selected model
            if algorithm == 'linear_regression':
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            elif algorithm == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression()
            elif algorithm == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier()
            elif algorithm == 'random_forest_regressor':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor()
            elif algorithm == 'decision_tree':
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier()
            elif algorithm == 'svm':
                from sklearn.svm import SVC
                model = SVC()
            elif algorithm == 'svm_regressor':
                from sklearn.svm import SVR
                model = SVR()
            elif algorithm == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier()
            elif algorithm == 'neural_network':
                from sklearn.neural_network import MLPClassifier
                model = MLPClassifier()
            else:
                return JsonResponse({'error': 'Invalid algorithm selected'})
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics based on problem type
            if problem_type == 'classification':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
                
                results = {
                    'problem_type': 'classification',
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm.tolist(),
                }
            else:  # regression
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results = {
                    'problem_type': 'regression',
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2,
                }
            
            # Return results
            return JsonResponse({
                'algorithm': algorithm.replace('_', ' ').title(),
                'problem_type': problem_type,
                **results
            })
            
        except Exception as e:
            return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
# def load_training_data(request):
#     """Load training data based on input method"""
#     if 'use_extracted' in request.POST:
#         # Use extracted features from session
#         csv_data = request.session.get('extracted_features', '')
#         if not csv_data:
#             return None, None, None, None
#         df = pd.read_csv(StringIO(csv_data))
#         X = df.iloc[:, :-1]  # Features
#         y = df.iloc[:, -1]   # Target
#         return train_test_split(X, y, test_size=0.2, random_state=42)
    
#     elif 'dataset_file' in request.FILES:
#         # Single file with split ratio
#         df = pd.read_csv(request.FILES['dataset_file'])
#         X = df.iloc[:, :-1]
#         y = df.iloc[:, -1]
#         test_size = int(request.POST.get('test_percent', 20)) / 100
#         return train_test_split(X, y, test_size=test_size, random_state=42)
    
#     elif 'training_file' in request.FILES and 'testing_file' in request.FILES:
#         # Separate training and testing files
#         train_df = pd.read_csv(request.FILES['training_file'])
#         test_df = pd.read_csv(request.FILES['testing_file'])
#         return (
#             train_df.iloc[:, :-1],  # X_train
#             test_df.iloc[:, :-1],   # X_test
#             train_df.iloc[:, -1],   # y_train
#             test_df.iloc[:, -1]     # y_test
#         )
    
#     return None, None, None, None

# def initialize_model(algorithm):
#     """Initialize the selected ML model"""
#     models = {
#         'random_forest': (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
#         'decision_tree': (DecisionTreeClassifier(random_state=42), "Decision Tree"),
#         'svm': (SVC(random_state=42, probability=True), "Support Vector Machine"),
#         'mlp': (MLPClassifier(random_state=42), "Multi-layer Perceptron"),
#         'knn': (KNeighborsClassifier(), "K-Nearest Neighbors"),
#         'logistic_regression': (LogisticRegression(random_state=42), "Logistic Regression"),
#         'lda': (LinearDiscriminantAnalysis(), "Linear Discriminant Analysis"),
#         'qda': (QuadraticDiscriminantAnalysis(), "Quadratic Discriminant Analysis"),
#         'sgd': (SGDClassifier(random_state=42), "Stochastic Gradient Descent"),
#         'naive_bayes': (GaussianNB(), "Naive Bayes"),
#         'bagging': (BaggingClassifier(random_state=42), "Bagging Classifier"),
#         'adaboost': (AdaBoostClassifier(random_state=42), "AdaBoost"),
#         'gbdt': (GradientBoostingClassifier(random_state=42), "Gradient Boosted Decision Trees")
#     }
    
#     # Add optional models if available
#     if XGBOOST_AVAILABLE and algorithm == 'xgboost':
#         models['xgboost'] = (XGBClassifier(random_state=42), "XGBoost")
#     if LIGHTGBM_AVAILABLE and algorithm == 'lightgbm':
#         models['lightgbm'] = (LGBMClassifier(random_state=42), "LightGBM")
    
#     if algorithm not in models:
#         raise ValueError(f"Invalid algorithm selected: {algorithm}")
    
#     return models[algorithm]

# def evaluate_model(model, X_test, y_test):
#     """Evaluate model performance"""
#     y_pred = model.predict(X_test)
#     return {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred, average='weighted'),
#         'recall': recall_score(y_test, y_pred, average='weighted'),
#         'f1_score': f1_score(y_test, y_pred, average='weighted')
#     }

# def save_model_to_disk(model, request):
#     """Save trained model to disk"""
#     model_dir = os.path.join(settings.MEDIA_ROOT, 'trained_models')
#     os.makedirs(model_dir, exist_ok=True)
#     model_path = os.path.join(model_dir, f'model_{request.session.session_key}.pkl')
#     joblib.dump(model, model_path)
#     return model_path

# def prepare_results(model_name, metrics, features, model_path):
#     """Prepare training results dictionary"""
#     return {
#         'model_name': model_name,
#         'accuracy': round(metrics['accuracy'], 4),
#         'precision': round(metrics['precision'], 4),
#         'recall': round(metrics['recall'], 4),
#         'f1_score': round(metrics['f1_score'], 4),
#         'features_used': list(features),
#         'model_path': model_path
#     }

# def training_results(request):
#     """Display training results"""
#     results = request.session.get('training_results')
#     if not results:
#         return redirect('module_selection')
#     return render(request, 'training_results.html', {'results': results})




def make_predictions(request):
    """Handle prediction requests"""
    if request.method == 'POST':
        try:
            # Load model
            model_path = os.path.join(
                settings.MEDIA_ROOT, 
                'trained_models', 
                f'model_{request.session.session_key}.pkl'
            )
            model = joblib.load(model_path)
            
            # Load and predict data
            df = pd.read_csv(request.FILES['prediction_data'])
            predictions = model.predict(df)
            
            # Prepare results
            df['prediction'] = predictions
            csv_output = df.to_csv(index=False)
            
            # Store results
            request.session['prediction_results'] = {
                'data': csv_output,
                'model_used': request.session.get('training_results', {}).get('model_name', 'Unknown')
            }
            
            return JsonResponse({
                'status': 'success',
                'predictions': predictions.tolist(),
                'download_url': '/download-predictions/'
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def data_visualization(request):
    return render(request, 'data_visualization.html')

def evaluation_values(request):
    return render(request, 'evaluation_values.html')