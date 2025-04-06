# Standard library imports
import os
import sys
import time
import json
import csv
from pathlib import Path
from io import StringIO, TextIOWrapper

# Django imports
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage
from django.conf import settings

# BioPython import
from Bio import SeqIO

# Data science imports
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, 
    precision_score, recall_score, 
    f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
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
import joblib

# Local imports
from Genitic_insight.forms import FastaUploadForm
from .utils.proteinfeature import ProteinFeatureExtractor
from .utils.RNAfeature import RNAFeatureExtractor
from .utils.DNAfeature import DNAFeatureExtractor
from django.http import JsonResponse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve, 
    mean_squared_error, mean_absolute_error, r2_score
)

from sklearn.preprocessing import LabelEncoder



# Optional ML packages
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent  # Adjust based on your structure
sys.path.append(str(project_root))
# Create your views here.



# views

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
    if not sequence:
        return "Unknown"
        
    # Clean and standardize the sequence
    sequence = sequence.upper().strip()
    sequence = ''.join([c for c in sequence if c.isalpha() or c in {'*', '-'}])
    
    if not sequence:
        return "Unknown"
    
    # Define character sets
    protein_chars = set("ACDEFGHIKLMNPQRSTVWY*")
    dna_chars = set("ATCG")
    rna_chars = set("AUCG")
    ambiguous_nuc = set("BDHKMNRSVWY-")
    
    # Get unique characters in sequence
    seq_chars = set(sequence)
    
    # Check for unambiguous protein characters
    protein_markers = protein_chars - dna_chars - rna_chars - ambiguous_nuc
    has_protein_markers = any(c in protein_markers for c in seq_chars)
    
    # Check for stop codon (protein marker)
    has_stop = '*' in seq_chars
    
    # Check for U (RNA) or T (DNA)
    has_u = 'U' in seq_chars
    has_t = 'T' in seq_chars
    
    # Detection logic
    if has_protein_markers or has_stop:
        return "Protein"
    
    if has_u and not has_t:
        if seq_chars.issubset(rna_chars | ambiguous_nuc):
            return "RNA"
    
    if has_t and not has_u:
        if seq_chars.issubset(dna_chars | ambiguous_nuc):
            return "DNA"
    
    # Handle ambiguous cases (no U/T and no protein markers)
    if not has_u and not has_t:
        if seq_chars.issubset(dna_chars | ambiguous_nuc):
            return "DNA"
        if seq_chars.issubset(protein_chars):
            return "Protein"
    
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
                    params=params_dict,
                    include_labels=True
                )
            elif sequence_type == 'DNA':
                extractor = DNAFeatureExtractor()
                csv_data = extractor.to_csv(
                    file_path,
                    methods=[descriptor],
                    params=params_dict,
                    include_labels=True
                )
            elif sequence_type == 'RNA':
                extractor = RNAFeatureExtractor()
                csv_data = extractor.to_csv(
                    file_path,
                    methods=[descriptor],
                    params=params_dict,
                    include_labels=True
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


def module_selection(request):
    """Render module selection page with available algorithms"""
    return render(request, 'module_selection.html')


@csrf_exempt
def train_model(request):
    if request.method == 'POST':
        try:
            # Validate file upload
            file_option = request.POST.get('file_option')
            if file_option == 'single' and 'dataset_file' not in request.FILES:
                return JsonResponse({'error': 'No dataset file provided'}, status=400)
            elif file_option == 'separate' and ('training_file' not in request.FILES or 'testing_file' not in request.FILES):
                return JsonResponse({'error': 'Both training and testing files are required'}, status=400)
            
            # Validate parameters
            try:
                train_percent = int(request.POST.get('train_percent', 80))
                if not (0 < train_percent < 100):
                    raise ValueError
            except (ValueError, TypeError):
                return JsonResponse({'error': 'Invalid train percentage (must be integer between 1-99)'}, status=400)
            
            algorithm = request.POST.get('algorithm')
            target_column = request.POST.get('target_column', 'label')  # Default to 'label' column
            
            # Process files
            try:
                if file_option == 'single':
                    df = pd.read_csv(request.FILES['dataset_file'])
                    if len(df.columns) < 2:
                        return JsonResponse({'error': 'Dataset must have at least 2 columns (features + target)'}, status=400)
                    
                    # Handle target column (either by name or index)
                    try:
                        # First try to convert to integer (for index)
                        target_col_idx = int(target_column)
                        if target_col_idx == -1:
                            target_col_idx = len(df.columns) - 1
                        if target_col_idx >= len(df.columns) or target_col_idx < 0:
                            return JsonResponse({'error': f'Target column index {target_col_idx} is out of range (0-{len(df.columns)-1})'}, status=400)
                        target_column_name = df.columns[target_col_idx]
                    except ValueError:
                        # If not an integer, treat as column name
                        if target_column not in df.columns:
                            return JsonResponse({'error': f'Target column "{target_column}" not found in dataset. Available columns: {list(df.columns)}'}, status=400)
                        target_column_name = target_column
                    
                    X = df.drop(target_column_name, axis=1)
                    y = df[target_column_name]
                    
                    # Convert y to numeric if it's categorical
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, train_size=train_percent/100, random_state=42, stratify=y
                    )
                else:  # separate files
                    train_df = pd.read_csv(request.FILES['training_file'])
                    test_df = pd.read_csv(request.FILES['testing_file'])
                    
                    if len(train_df.columns) < 2 or len(test_df.columns) < 2:
                        return JsonResponse({'error': 'Files must have at least 2 columns (features + target)'}, status=400)
                    
                    # Handle target column for separate files
                    try:
                        # First try to convert to integer (for index)
                        target_col_idx = int(target_column)
                        if target_col_idx == -1:
                            target_col_idx = len(train_df.columns) - 1
                        if target_col_idx >= len(train_df.columns) or target_col_idx < 0:
                            return JsonResponse({'error': f'Target column index {target_col_idx} is out of range (0-{len(train_df.columns)-1})'}, status=400)
                        target_column_name = train_df.columns[target_col_idx]
                    except ValueError:
                        # If not an integer, treat as column name
                        if target_column not in train_df.columns or target_column not in test_df.columns:
                            return JsonResponse({'error': f'Target column "{target_column}" not found in both files'}, status=400)
                        target_column_name = target_column
                    
                    X_train = train_df.drop(target_column_name, axis=1)
                    y_train = train_df[target_column_name]
                    X_test = test_df.drop(target_column_name, axis=1)
                    y_test = test_df[target_column_name]
                    
                    # Convert y to numeric if it's categorical
                    if y_train.dtype == 'object':
                        le = LabelEncoder()
                        y_train = le.fit_transform(y_train)
                        y_test = le.transform(y_test)
            except Exception as e:
                return JsonResponse({'error': f'Error processing data: {str(e)}'}, status=400)
            
            # Determine problem type
            unique_classes = np.unique(y_train)
            if len(unique_classes) <= 10 or y_train.dtype == 'object':
                problem_type = 'classification'
                # For binary classification, we need exactly 2 classes
                if len(unique_classes) == 2:
                    binary_classification = True
                else:
                    binary_classification = False
            else:
                problem_type = 'regression'
                binary_classification = False
            
            # Validate algorithm based on problem type
            regression_algorithms = ['linear_regression', 'random_forest_regressor', 'svm_regressor']
            classification_algorithms = ['logistic_regression', 'random_forest', 'decision_tree', 'svm', 'knn', 'neural_network']
            
            if problem_type == 'regression' and algorithm not in regression_algorithms:
                return JsonResponse({'error': f'Selected algorithm is not suitable for regression problems. Please use: {", ".join(regression_algorithms)}'}, status=400)
            elif problem_type == 'classification' and algorithm not in classification_algorithms:
                return JsonResponse({'error': f'Selected algorithm is not suitable for classification problems. Please use: {", ".join(classification_algorithms)}'}, status=400)
            
            # Train selected model
            try:
                if algorithm == 'linear_regression':
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                elif algorithm == 'logistic_regression':
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(max_iter=1000)
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
                    model = SVC(probability=True)
                elif algorithm == 'svm_regressor':
                    from sklearn.svm import SVR
                    model = SVR()
                elif algorithm == 'knn':
                    from sklearn.neighbors import KNeighborsClassifier
                    model = KNeighborsClassifier()
                elif algorithm == 'neural_network':
                    from sklearn.neural_network import MLPClassifier
                    model = MLPClassifier(max_iter=1000)
                else:
                    return JsonResponse({'error': 'Invalid algorithm selected'})
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            except Exception as e:
                return JsonResponse({'error': f'Error training model: {str(e)}'}, status=400)
            
            # Calculate metrics based on problem type
            results = {}
            if problem_type == 'classification':
                # Calculate basic classification metrics
                cm = confusion_matrix(y_test, y_pred)
                
                # For binary classification, we can calculate ROC/AUC
                roc_data = None
                pr_data = None
                if binary_classification:
                    try:
                        # Get probability scores for ROC curve
                        if hasattr(model, 'predict_proba'):
                            y_scores = model.predict_proba(X_test)[:, 1]
                        else:  # For models that don't do probability estimates
                            y_scores = model.decision_function(X_test)
                            y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
                        
                        # Calculate ROC curve and AUC
                        fpr, tpr, _ = roc_curve(y_test, y_scores)
                        roc_auc = auc(fpr, tpr)
                        
                        # Calculate Precision-Recall curve and AUC
                        precision, recall, _ = precision_recall_curve(y_test, y_scores)
                        pr_auc = auc(recall, precision)
                        
                        roc_data = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'auc': roc_auc
                        }
                        
                        pr_data = {
                            'precision': precision.tolist(),
                            'recall': recall.tolist(),
                            'auprc': pr_auc
                        }
                    except Exception as e:
                        # If ROC calculation fails, continue without it
                        pass
                
                # Calculate classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision_score_val = precision_score(y_test, y_pred, average='weighted')
                recall_score_val = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results = {
                    'problem_type': 'classification',
                    'accuracy': accuracy,
                    'precision': precision_score_val,
                    'recall': recall_score_val,
                    'f1_score': f1,
                    'confusion_matrix': cm.tolist(),
                    'classes': unique_classes.tolist(),
                    'binary_classification': binary_classification,
                }
                
                if roc_data:
                    results['roc_curve'] = roc_data
                if pr_data:
                    results['pr_curve'] = pr_data
                
            else:  # regression
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
                'target_column': target_column_name,
                **results
            })
            
        except Exception as e:
            return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)


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