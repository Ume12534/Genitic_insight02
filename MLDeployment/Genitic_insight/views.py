# Standard library imports
import os
import sys
import time
import json
import csv
from pathlib import Path
from io import StringIO, TextIOWrapper
import uuid
import tempfile
from venv import logger

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
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, mean_squared_error, mean_absolute_error,
    r2_score
)
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

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
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


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

# Global dictionary to store extracted feature data between requests
EXTRACTED_DATA_STORE = {}


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

def feature_extraction(request):
    if request.method == 'POST' and request.FILES.get('fasta_file'):
        try:
            # Use TextIOWrapper in a 'with' block to ensure it closes
            with TextIOWrapper(request.FILES['fasta_file'].file, encoding='utf-8') as fasta_file:
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
    """
    Handle sequence feature extraction request and manage extracted CSV data storage.
    Automatically replaces old extracted data with new data in 'extracted_CSV.csv'.
    
    Args:
        request: Django HTTP request object containing:
            - Fasta file upload
            - Analysis parameters (descriptor, sequence type, etc.)
    
    Returns:
        JsonResponse containing:
            - Success message or error
            - Processed CSV data without ID column
            - Metadata about the analysis
    """
    # Define the path for storing the extracted CSV
    EXTRACTED_CSV_PATH = os.path.join(settings.MEDIA_ROOT, 'extracted_CSV.csv')
    
    if request.method == 'POST' and request.FILES.get('fasta_file'):
        # Initialize file paths for cleanup
        input_file_path = None
        
        try:
            # =============================================
            # 1. GET REQUEST PARAMETERS AND VALIDATE INPUTS
            # =============================================
            fasta_file = request.FILES['fasta_file']
            descriptor = request.POST.get('descriptor', '')
            sequence_type = request.POST.get('sequence_type', 'Unknown')
            parameters_str = request.POST.get('parameter', '') 
            
            # =============================================
            # 2. SETUP TEMPORARY STORAGE LOCATION
            # =============================================
            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Clean up any previous temporary files
            for f in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except (PermissionError, IsADirectoryError):
                    continue  # Skip files that can't be deleted
            
            # =============================================
            # 3. SAVE UPLOADED FILE TO TEMP LOCATION
            # =============================================
            # Generate unique filename to prevent collisions
            unique_id = uuid.uuid4().hex
            input_file_path = os.path.join(temp_dir, f"{unique_id}_{fasta_file.name}")
            
            # Safely save the uploaded file with error handling
            try:
                with open(input_file_path, 'wb+') as destination:
                    for chunk in fasta_file.chunks():
                        destination.write(chunk)
            except IOError as e:
                return JsonResponse({
                    'error': 'File save failed',
                    'details': str(e)
                }, status=500)
            
            # =============================================
            # 4. PROCESS PARAMETERS
            # =============================================
            params_dict = {}
            if parameters_str:
                try:
                    # Convert parameters string (key=value,key2=value2) to dictionary
                    params_dict = dict(item.split('=') for item in parameters_str.split(','))
                    
                    # Convert numeric parameters to appropriate types
                    for key, value in params_dict.items():
                        try:
                            params_dict[key] = float(value) if '.' in value else int(value)
                        except ValueError:
                            pass  # Keep as string if conversion fails
                except Exception as e:
                    print(f"Parameter parsing error: {e}")
                    params_dict = {}  # Use empty dict if parsing fails
            
            # =============================================
            # 5. PERFORM FEATURE EXTRACTION
            # =============================================
            csv_data = None
            
            try:
                if sequence_type == 'Protein':
                    extractor = ProteinFeatureExtractor()
                    csv_data = extractor.to_csv(
                        input_file_path,
                        methods=[descriptor],
                        params=params_dict,
                        include_labels=True
                    )
                elif sequence_type == 'DNA':
                    extractor = DNAFeatureExtractor()
                    csv_data = extractor.to_csv(
                        input_file_path,
                        methods=[descriptor],
                        params=params_dict,
                        include_labels=True
                    )
                elif sequence_type == 'RNA':
                    extractor = RNAFeatureExtractor()
                    csv_data = extractor.to_csv(
                        input_file_path,
                        methods=[descriptor],
                        params=params_dict,
                        include_labels=True
                    )
                else:
                    raise ValueError(f"Unknown sequence type: {sequence_type}")
            except Exception as e:
                return JsonResponse({
                    'error': 'Feature extraction failed',
                    'details': str(e)
                }, status=500)
            
            # =============================================
            # 6. PROCESS CSV DATA (REMOVE ID COLUMN)
            # =============================================
            if csv_data:
                lines = csv_data.split('\n')
                if lines:  # Check if we have any data
                    headers = lines[0].split(',')
                    
                    # Find and remove ID column if it exists
                    if 'ID' in headers:
                        id_index = headers.index('ID')
                        headers.pop(id_index)  # Remove from header
                        
                        # Process all data rows
                        processed_lines = []
                        for line in lines:
                            if not line.strip():  # Skip empty lines
                                continue
                            cells = line.split(',')
                            if len(cells) > id_index:  # Make sure row has enough columns
                                cells.pop(id_index)  # Remove ID value
                            processed_lines.append(','.join(cells))
                        
                        # Rebuild CSV without ID column
                        csv_data = '\n'.join(processed_lines)
            
            # =============================================
            # 7. STORE EXTRACTED DATA IN extracted_CSV.csv
            # =============================================
            if csv_data:
                # Remove old extracted CSV file if it exists
                try:
                    if os.path.exists(EXTRACTED_CSV_PATH):
                        os.remove(EXTRACTED_CSV_PATH)
                except Exception as e:
                    print(f"Warning: Could not delete old extracted CSV: {e}")
                
                # Save new extracted data to extracted_CSV.csv
                try:
                    with open(EXTRACTED_CSV_PATH, 'w') as f:
                        f.write(csv_data)
                except IOError as e:
                    return JsonResponse({
                        'error': 'Failed to save extracted features',
                        'details': str(e)
                    }, status=500)
            
            # =============================================
            # 8. PREPARE RESPONSE
            # =============================================
            response_data = {
                'message': 'Feature extraction completed',
                'file_name': fasta_file.name,
                'sequence_type': sequence_type,
                'descriptor': descriptor,
                'csv_data': csv_data,  # This now has no ID column
                'extracted_csv_path': EXTRACTED_CSV_PATH  # Path to stored CSV
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            # =============================================
            # ERROR HANDLING AND CLEANUP
            # =============================================
            # Clean up any temporary files if they exist
            try:
                if input_file_path and os.path.exists(input_file_path):
                    os.remove(input_file_path)
            except:
                pass
                
            return JsonResponse({
                'error': 'Processing failed',
                'details': str(e)
            }, status=400)
    
    # Return error for non-POST requests or missing file
    return JsonResponse({
        'error': 'Invalid request',
        'details': 'Only POST requests with file upload are accepted'
    }, status=400)

def cleanup_temp_files():
    """
    Utility function to clean up all temporary files.
    Can be called periodically or when needed.
    """
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    if os.path.exists(temp_dir):
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                try:
                    os.remove(os.path.join(root, file))
                except Exception as e:
                    print(f"Error deleting temp file {file}: {e}")

def module_selection(request):
    """Render module selection page with available algorithms"""
    return render(request, 'module_selection.html')


@csrf_exempt
def train_model(request):
    """
    Handles model training requests from the frontend.
    Supports three data sources:
    1. Uploaded dataset file with train/test split
    2. Separate training and testing files
    3. Extracted feature data from analyze_sequence()
    
    Args:
        request: Django HTTP request containing:
            - Data source specification (file_option)
            - Multiple algorithm selections with parameters
            - For extracted data: extraction_id
    
    Returns:
        JsonResponse with training results or error message
    """
    if request.method == 'POST':
        try:
            # ========== REQUEST VALIDATION ==========
            file_option = request.POST.get('file_option')
            extracted_data = request.POST.get('extracted_data')
            
            # Validate data source
            if not extracted_data:
                if file_option == 'single' and 'dataset_file' not in request.FILES:
                    return JsonResponse({'error': 'No dataset file provided'}, status=400)
                elif file_option == 'separate' and ('training_file' not in request.FILES or 'testing_file' not in request.FILES):
                    return JsonResponse({'error': 'Both training and testing files are required'}, status=400)
                
            # ========== PARAMETER VALIDATION ==========
            try:
                train_percent = int(request.POST.get('train_percent', 80))
                if not (0 < train_percent < 100):
                    raise ValueError
            except (ValueError, TypeError):
                return JsonResponse({'error': 'Invalid train percentage (must be integer between 1-99)'}, status=400)
            
            # Parse selected algorithms and their parameters
            try:
                selected_algorithms = json.loads(request.POST.get('algorithms', '[]'))
                if not selected_algorithms:
                    return JsonResponse({'error': 'No algorithms selected'}, status=400)
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid algorithm parameters'}, status=400)
                
            target_column = request.POST.get('target_column', 'label')
            
            # ========== DATA PROCESSING ==========
            try:
                
                if extracted_data:

                    EXTRACTED_CSV_PATH = os.path.join(settings.MEDIA_ROOT, 'extracted_CSV.csv')
    
                    if not os.path.exists(EXTRACTED_CSV_PATH):
                        return JsonResponse({
                            'success': False,
                            'error': 'No extracted data available',
                            'details': 'Please perform feature extraction first'
                        }, status=404)
                

                    # # Get extraction_id from session
                    # extraction_id = request.session.get('extracted_features', {}).get('extraction_id')
                   
                    # if not extraction_id:
                    #     return JsonResponse({'error': 'No extraction ID found in session'}, status=400)

                    # # Construct the file path from extraction ID
                    # temp_output_dir = os.path.join(settings.MEDIA_ROOT, 'temp', 'output')
                    # features_file_path = os.path.join(temp_output_dir, f"{extraction_id}_features.csv")

                    # # Check if file exists
                    # if not os.path.exists(features_file_path):
                    #     return JsonResponse({'error': f'No features file found with ID {extraction_id}'}, status=400)
                    
                    # Read CSV into DataFrame
                    df = pd.read_csv(EXTRACTED_CSV_PATH)
                    if len(df.columns) < 2:
                        return JsonResponse({'error': 'Dataset must have at least 2 columns (features + target)'}, status=400)
                    
                    # Handle target column selection
                    try:
                        target_col_idx = int(target_column)
                        if target_col_idx == -1:
                            target_col_idx = len(df.columns) - 1
                        if target_col_idx >= len(df.columns) or target_col_idx < 0:
                            return JsonResponse({'error': f'Target column index {target_col_idx} is out of range (0-{len(df.columns)-1})'}, status=400)
                        target_column_name = df.columns[target_col_idx]
                    except ValueError:
                        if target_column not in df.columns:
                            return JsonResponse({'error': f'Target column "{target_column}" not found in dataset. Available columns: {list(df.columns)}'}, status=400)
                        target_column_name = target_column
                    
                    X = df.drop(target_column_name, axis=1)
                    y = df[target_column_name]
                    
                    # Encode categorical target if needed
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                    
                    # Split data into train and test sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        train_size=train_percent/100, 
                        random_state=42, 
                        stratify=y if len(np.unique(y)) > 1 else None
                    )
                    
                elif file_option == 'single':
                    # Process single file with train/test split
                    df = pd.read_csv(request.FILES['dataset_file'])
                    if len(df.columns) < 2:
                        return JsonResponse({'error': 'Dataset must have at least 2 columns (features + target)'}, status=400)
                    
                    # Handle target column selection
                    try:
                        target_col_idx = int(target_column)
                        if target_col_idx == -1:
                            target_col_idx = len(df.columns) - 1
                        if target_col_idx >= len(df.columns) or target_col_idx < 0:
                            return JsonResponse({'error': f'Target column index {target_col_idx} is out of range (0-{len(df.columns)-1})'}, status=400)
                        target_column_name = df.columns[target_col_idx]
                    except ValueError:
                        if target_column not in df.columns:
                            return JsonResponse({'error': f'Target column "{target_column}" not found in dataset. Available columns: {list(df.columns)}'}, status=400)
                        target_column_name = target_column
                    
                    X = df.drop(target_column_name, axis=1)
                    y = df[target_column_name]
                    
                    # Encode categorical target if needed
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                    
                    # Split data into train and test sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        train_size=train_percent/100, 
                        random_state=42, 
                        stratify=y if len(np.unique(y)) > 1 else None
                    )
                else:  # separate files
                    # Process separate training and testing files
                    train_df = pd.read_csv(request.FILES['training_file'])
                    test_df = pd.read_csv(request.FILES['testing_file'])
                    
                    if len(train_df.columns) < 2 or len(test_df.columns) < 2:
                        return JsonResponse({'error': 'Files must have at least 2 columns (features + target)'}, status=400)
                    
                    # Handle target column selection for separate files
                    try:
                        target_col_idx = int(target_column)
                        if target_col_idx == -1:
                            target_col_idx = len(train_df.columns) - 1
                        if target_col_idx >= len(train_df.columns) or target_col_idx < 0:
                            return JsonResponse({'error': f'Target column index {target_col_idx} is out of range (0-{len(train_df.columns)-1})'}, status=400)
                        target_column_name = train_df.columns[target_col_idx]
                    except ValueError:
                        if target_column not in train_df.columns or target_column not in test_df.columns:
                            return JsonResponse({'error': f'Target column "{target_column}" not found in both files'}, status=400)
                        target_column_name = target_column
                    
                    X_train = train_df.drop(target_column_name, axis=1)
                    y_train = train_df[target_column_name]
                    X_test = test_df.drop(target_column_name, axis=1)
                    y_test = test_df[target_column_name]
                    
                    # Encode categorical target if needed
                    if y_train.dtype == 'object':
                        le = LabelEncoder()
                        y_train = le.fit_transform(y_train)
                        y_test = le.transform(y_test)
            except Exception as e:
                return JsonResponse({'error': f'Error processing data: {str(e)}'}, status=400)
            
            # ========== PROBLEM TYPE DETECTION ==========
            unique_classes = np.unique(y_train)
            if len(unique_classes) <= 10 or y_train.dtype == 'object':
                problem_type = 'classification'
                binary_classification = len(unique_classes) == 2
            else:
                problem_type = 'regression'
                binary_classification = False
            
            # ========== TRAIN ALL SELECTED ALGORITHMS ==========
            results = []
            
            for algo_config in selected_algorithms:
                algorithm = algo_config['algorithm']
                parameters = algo_config.get('parameters', {})
                
                try:
                    # Validate algorithm against problem type
                    regression_algorithms = ['linear_regression', 'random_forest_regressor', 'svm_regressor']
                    classification_algorithms = ['logistic_regression', 'random_forest', 'decision_tree', 'svm', 'knn', 'neural_network']
                    
                    if problem_type == 'regression' and algorithm not in regression_algorithms:
                        results.append({
                            'algorithm': algorithm,
                            'error': f'Algorithm not suitable for regression problems'
                        })
                        continue
                    elif problem_type == 'classification' and algorithm not in classification_algorithms:
                        results.append({
                            'algorithm': algorithm,
                            'error': f'Algorithm not suitable for classification problems'
                        })
                        continue
                    
                    # Initialize appropriate model based on selected algorithm with parameters
                    model = None
                    if algorithm == 'linear_regression':
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression(
                            fit_intercept=parameters.get('fit_intercept', True),
                            normalize=parameters.get('normalize', False)
                        )
                    elif algorithm == 'logistic_regression':
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(
                            C=parameters.get('C', 1.0),
                            max_iter=parameters.get('max_iter', 100),
                            random_state=42
                        )
                    elif algorithm == 'random_forest':
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(
                            n_estimators=parameters.get('n_estimators', 100),
                            max_depth=parameters.get('max_depth', None),
                            min_samples_split=parameters.get('min_samples_split', 2),
                            random_state=42
                        )
                    elif algorithm == 'random_forest_regressor':
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(
                            n_estimators=parameters.get('n_estimators', 100),
                            max_depth=parameters.get('max_depth', None),
                            min_samples_split=parameters.get('min_samples_split', 2),
                            random_state=42
                        )
                    elif algorithm == 'decision_tree':
                        from sklearn.tree import DecisionTreeClassifier
                        model = DecisionTreeClassifier(
                            max_depth=parameters.get('max_depth', None),
                            min_samples_split=parameters.get('min_samples_split', 2),
                            random_state=42
                        )
                    elif algorithm == 'svm':
                        from sklearn.svm import SVC
                        model = SVC(
                            C=parameters.get('C', 1.0),
                            kernel=parameters.get('kernel', 'rbf'),
                            probability=True,
                            random_state=42
                        )
                    elif algorithm == 'svm_regressor':
                        from sklearn.svm import SVR
                        model = SVR(
                            C=parameters.get('C', 1.0),
                            kernel=parameters.get('kernel', 'rbf')
                        )
                    elif algorithm == 'knn':
                        from sklearn.neighbors import KNeighborsClassifier
                        model = KNeighborsClassifier(
                            n_neighbors=parameters.get('n_neighbors', 5),
                            weights=parameters.get('weights', 'uniform')
                        )
                    elif algorithm == 'neural_network':
                        from sklearn.neural_network import MLPClassifier
                        # Parse hidden layer sizes from string (e.g. "100,50" -> (100, 50))
                        hidden_layer_sizes = tuple(
                            int(x) for x in parameters.get('hidden_layer_sizes', '100').split(',')
                        ) if parameters.get('hidden_layer_sizes') else (100,)
                        model = MLPClassifier(
                            hidden_layer_sizes=hidden_layer_sizes,
                            activation=parameters.get('activation', 'relu'),
                            learning_rate=parameters.get('learning_rate', 'constant'),
                            max_iter=1000,
                            random_state=42
                        )
                    else:
                        results.append({
                            'algorithm': algorithm,
                            'error': 'Invalid algorithm selected'
                        })
                        continue
                    
                    # Train the model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # For classification, get probability scores if available
                    y_scores = None
                    if problem_type == 'classification':
                        try:
                            if hasattr(model, 'predict_proba'):
                                y_scores = model.predict_proba(X_test)[:, 1]
                            elif hasattr(model, 'decision_function'):
                                y_scores = model.decision_function(X_test)
                        except:
                            pass
                    
                    # Prepare result object
                    algo_result = {
                        'algorithm': algorithm.replace('_', ' ').title(),
                        'parameters': parameters
                    }
                    
                    if problem_type == 'classification':
                        # Calculate classification metrics
                        cm = confusion_matrix(y_test, y_pred)
                        accuracy = accuracy_score(y_test, y_pred)
                        precision_score_val = precision_score(y_test, y_pred, average='weighted')
                        recall_score_val = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                        algo_result.update({
                            'accuracy': accuracy,
                            'precision': precision_score_val,
                            'recall': recall_score_val,
                            'f1_score': f1,
                            'confusion_matrix': cm.tolist(),
                            'classes': unique_classes.tolist(),
                        })
                        
                        # ROC and PRC for binary classification
                        if binary_classification and y_scores is not None:
                            try:
                                # Calculate ROC curve
                                fpr, tpr, _ = roc_curve(y_test, y_scores)
                                roc_auc = auc(fpr, tpr)
                                
                                # Calculate Precision-Recall curve
                                precision, recall, _ = precision_recall_curve(y_test, y_scores)
                                pr_auc = average_precision_score(y_test, y_scores)
                                
                                algo_result.update({
                                    'roc_curve': {
                                        'fpr': fpr.tolist(),
                                        'tpr': tpr.tolist(),
                                        'auc': roc_auc
                                    },
                                    'pr_curve': {
                                        'recall': recall.tolist(),
                                        'precision': precision.tolist(),
                                        'auprc': pr_auc
                                    }
                                })
                            except Exception as e:
                                print(f"Error generating curves for {algorithm}: {str(e)}")
                                
                    else:  # regression
                        # Calculate regression metrics
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        algo_result.update({
                            'mse': mse,
                            'mae': mae,
                            'r2_score': r2,
                        })
                    
                    results.append(algo_result)
                    
                except Exception as e:
                    results.append({
                        'algorithm': algorithm,
                        'error': f'Training failed: {str(e)}'
                    })
                    continue
            
            # ========== RETURN RESULTS ==========
            return JsonResponse({
                'status': 'success',
                'problem_type': problem_type,
                'target_column': target_column_name,
                'binary_classification': binary_classification,
                'results': results,
                'data_source': 'extracted' if extracted_data else 'uploaded'
            })
            
        except Exception as e:
            return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)



def module_selection_with_features(request):
    """
    Render module selection page with option to train on extracted features.
    """
    extraction_id = request.GET.get('extraction_id')
    extracted_data = None
    
    if extraction_id and extraction_id in EXTRACTED_DATA_STORE:
        data_info = EXTRACTED_DATA_STORE[extraction_id]
        try:
            # Read the extracted features file
            with open(data_info['file_path'], 'r') as f:
                extracted_data = f.read()
            
            # Get target column (assuming last column is target)
            df = pd.read_csv(StringIO(extracted_data))
            target_column = df.columns[-1]
            
            context = {
                'preloaded_features': True,
                'extraction_id': extraction_id,
                'target_column': target_column,
                'sequence_type': data_info['sequence_type'],
                'descriptor': data_info['descriptor'],
                'features_preview': extracted_data.split('\n')[:10]  # First 10 lines for preview
            }
            return render(request, 'module_selection_with_features.html', context)
            
        except Exception as e:
            # Clean up if file is corrupted
            if os.path.exists(data_info['file_path']):
                os.remove(data_info['file_path'])
            del EXTRACTED_DATA_STORE[extraction_id]
    
    # Default case (no valid extracted data)
    return render(request, 'module_selection_with_features.html', {
        'preloaded_features': False,
        'error': 'No valid extracted features found'
    })

def train_model_with_features(request):
    if request.method == 'POST':
        try:
            # Get the algorithm and parameters from the form
            algorithm = request.POST.get('algorithm')
            parameters = request.POST.get('parameters', '')
            
            # Get the features data from session
            csv_data = request.session.get('extracted_features', '')
            
            if not csv_data:
                return JsonResponse({'error': 'No features data found'}, status=400)
            
            # Here you would implement your actual model training logic
            # This is just a placeholder for the response
            response_data = {
                'message': 'Model training started',
                'algorithm': algorithm,
                'status': 'success'
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)


def data_visualization(request):
    return render(request, 'data_visualization.html')

def evaluation_values(request):
    return render(request, 'evaluation_values.html')

def about_view(request):
    return render(request, 'about.html')




@csrf_exempt
def generate_model_report(request):
    """
    Generates a downloadable report of model training results.
    Supports PDF, HTML, and CSV formats.
    
    Args:
        request: Django HTTP request containing:
            - model_data: The training results data
            - format: Report format (pdf, html, csv)
            - report_name: Base name for the report file
            - options: What to include in the report
            
    Returns:
        File response with the generated report
    """
    if request.method == 'POST':
        try:
            # Parse request data
            data = json.loads(request.body)
            model_data = data.get('model_data')
            report_format = data.get('format', 'pdf')
            report_name = data.get('report_name', 'Model_Analysis_Report')
            options = data.get('options', {})
            
            if not model_data:
                return JsonResponse({'error': 'No model data provided'}, status=400)
            
            # Create temporary directory for report assets
            with tempfile.TemporaryDirectory() as temp_dir:
                if report_format == 'pdf':
                    return generate_pdf_report(model_data, report_name, options, temp_dir)
                elif report_format == 'html':
                    return generate_html_report(model_data, report_name, options, temp_dir)
                elif report_format == 'csv':
                    return generate_csv_report(model_data, report_name, options)
                else:
                    return JsonResponse({'error': 'Invalid report format'}, status=400)
                    
        except Exception as e:
            return JsonResponse({'error': f'Report generation failed: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)

def generate_pdf_report(model_data, report_name, options, temp_dir):
    """
    Generates a PDF report of model training results.
    
    Args:
        model_data: Dictionary containing model training results
        report_name: Base name for the report file
        options: Dictionary of report options
        temp_dir: Temporary directory for storing assets
        
    Returns:
        FileResponse with the PDF report
    """
    try:
        # Create buffer for PDF
        buffer = BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Add title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            alignment=1,
            spaceAfter=20
        )
        elements.append(Paragraph('Model Training Analysis Report', title_style))
        
        # Add metadata
        meta_style = styles['Normal']
        elements.append(Paragraph(f'<b>Problem Type:</b> {model_data.get("problem_type", "N/A")}', meta_style))
        elements.append(Paragraph(f'<b>Target Column:</b> {model_data.get("target_column", "N/A")}', meta_style))
        elements.append(Spacer(1, 20))
        
        # Add metrics section if requested
        if options.get('include_metrics', True):
            elements.append(Paragraph('<b>Model Performance Metrics</b>', styles['Heading2']))
            
            # Create metrics table
            results = model_data.get('results', [])
            if results:
                # Prepare table data
                table_data = [['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1 Score']]
                
                for result in results:
                    if 'error' in result:
                        table_data.append([result['algorithm'], result['error'], '', '', ''])
                    else:
                        table_data.append([
                            result['algorithm'],
                            f"{result.get('accuracy', 'N/A'):.4f}" if 'accuracy' in result else 'N/A',
                            f"{result.get('precision', 'N/A'):.4f}" if 'precision' in result else 'N/A',
                            f"{result.get('recall', 'N/A'):.4f}" if 'recall' in result else 'N/A',
                            f"{result.get('f1_score', 'N/A'):.4f}" if 'f1_score' in result else 'N/A'
                        ])
                
                # Create and style table
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                elements.append(Spacer(1, 20))
        
        # Add parameters section if requested
        if options.get('include_parameters', True):
            elements.append(Paragraph('<b>Algorithm Parameters</b>', styles['Heading2']))
            
            for result in model_data.get('results', []):
                if 'parameters' in result and result['parameters']:
                    elements.append(Paragraph(f"<b>{result['algorithm']}</b>", styles['Normal']))
                    
                    param_data = []
                    for param, value in result['parameters'].items():
                        param_data.append([param, str(value)])
                    
                    param_table = Table(param_data, colWidths=[200, 200])
                    param_table.setStyle(TableStyle([
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP')
                    ]))
                    elements.append(param_table)
                    elements.append(Spacer(1, 10))
            
            elements.append(Spacer(1, 20))
        
        # Add charts section if requested and available
        if options.get('include_charts', True) and model_data.get('binary_classification', False):
            elements.append(Paragraph('<b>Performance Visualizations</b>', styles['Heading2']))
            
            # Generate ROC curve image
            roc_fig = generate_roc_curve_figure(model_data)
            if roc_fig:
                roc_img_path = os.path.join(temp_dir, 'roc_curve.png')
                roc_fig.savefig(roc_img_path, bbox_inches='tight', dpi=300)
                plt.close(roc_fig)
                
                elements.append(Paragraph('ROC Curves', styles['Heading3']))
                elements.append(Image(roc_img_path, width=400, height=300))
                elements.append(Spacer(1, 20))
            
            # Generate PR curve image
            pr_fig = generate_pr_curve_figure(model_data)
            if pr_fig:
                pr_img_path = os.path.join(temp_dir, 'pr_curve.png')
                pr_fig.savefig(pr_img_path, bbox_inches='tight', dpi=300)
                plt.close(pr_fig)
                
                elements.append(Paragraph('Precision-Recall Curves', styles['Heading3']))
                elements.append(Image(pr_img_path, width=400, height=300))
                elements.append(Spacer(1, 20))
        
        # Build PDF document
        doc.build(elements)
        
        # Prepare response
        buffer.seek(0)
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{report_name}.pdf"'
        return response
        
    except Exception as e:
        return JsonResponse({'error': f'PDF generation failed: {str(e)}'}, status=500)

def generate_roc_curve_figure(model_data):
    """
    Generates a matplotlib figure for ROC curves from model data.
    
    Args:
        model_data: Dictionary containing model training results
        
    Returns:
        Matplotlib figure object or None if no ROC data available
    """
    if not model_data.get('binary_classification', False):
        return None
    
    results_with_roc = [r for r in model_data.get('results', []) if 'roc_curve' in r]
    if not results_with_roc:
        return None
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, result in enumerate(results_with_roc):
        roc_data = result['roc_curve']
        plt.plot(
            roc_data['fpr'], 
            roc_data['tpr'], 
            color=colors[i % len(colors)],
            label=f"{result['algorithm']} (AUC = {roc_data['auc']:.2f})"
        )
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    return plt.gcf()

def generate_pr_curve_figure(model_data):
    """
    Generates a matplotlib figure for Precision-Recall curves from model data.
    
    Args:
        model_data: Dictionary containing model training results
        
    Returns:
        Matplotlib figure object or None if no PR data available
    """
    if not model_data.get('binary_classification', False):
        return None
    
    results_with_pr = [r for r in model_data.get('results', []) if 'pr_curve' in r]
    if not results_with_pr:
        return None
    
    plt.figure(figsize=(8, 6))
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, result in enumerate(results_with_pr):
        pr_data = result['pr_curve']
        plt.plot(
            pr_data['recall'], 
            pr_data['precision'], 
            color=colors[i % len(colors)],
            label=f"{result['algorithm']} (AP = {pr_data['auprc']:.2f})"
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid(True)
    
    return plt.gcf()

def generate_html_report(model_data, report_name, options, temp_dir):
    """
    Generates an HTML report of model training results.
    
    Args:
        model_data: Dictionary containing model training results
        report_name: Base name for the report file
        options: Dictionary of report options
        temp_dir: Temporary directory for storing assets
        
    Returns:
        FileResponse with the HTML report
    """
    try:
        # Start building HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                h3 {{ color: #7f8c8d; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-table {{ width: 80%; margin: 20px auto; }}
                .chart {{ text-align: center; margin: 30px 0; }}
                .chart img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Model Training Analysis Report</h1>
            <p><strong>Problem Type:</strong> {model_data.get('problem_type', 'N/A')}</p>
            <p><strong>Target Column:</strong> {model_data.get('target_column', 'N/A')}</p>
        """
        
        # Add metrics section if requested
        if options.get('include_metrics', True):
            html_content += "<h2>Model Performance Metrics</h2>"
            
            results = model_data.get('results', [])
            if results:
                html_content += """
                <table class="metric-table">
                    <tr>
                        <th>Algorithm</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                    </tr>
                """
                
                for result in results:
                    if 'error' in result:
                        html_content += f"""
                        <tr>
                            <td>{result['algorithm']}</td>
                            <td colspan="4">{result['error']}</td>
                        </tr>
                        """
                    else:
                        html_content += f"""
                        <tr>
                            <td>{result['algorithm']}</td>
                            <td>{result.get('accuracy', 'N/A'):.4f if 'accuracy' in result else 'N/A'}</td>
                            <td>{result.get('precision', 'N/A'):.4f if 'precision' in result else 'N/A'}</td>
                            <td>{result.get('recall', 'N/A'):.4f if 'recall' in result else 'N/A'}</td>
                            <td>{result.get('f1_score', 'N/A'):.4f if 'f1_score' in result else 'N/A'}</td>
                        </tr>
                        """
                
                html_content += "</table>"
        
        # Add parameters section if requested
        if options.get('include_parameters', True):
            html_content += "<h2>Algorithm Parameters</h2>"
            
            for result in model_data.get('results', []):
                if 'parameters' in result and result['parameters']:
                    html_content += f"<h3>{result['algorithm']}</h3><table>"
                    
                    for param, value in result['parameters'].items():
                        html_content += f"""
                        <tr>
                            <td><strong>{param}</strong></td>
                            <td>{value}</td>
                        </tr>
                        """
                    
                    html_content += "</table>"
        
        # Add charts section if requested and available
        if options.get('include_charts', True) and model_data.get('binary_classification', False):
            html_content += "<h2>Performance Visualizations</h2>"
            
            # Generate ROC curve image
            roc_fig = generate_roc_curve_figure(model_data)
            if roc_fig:
                roc_img_path = os.path.join(temp_dir, 'roc_curve.png')
                roc_fig.savefig(roc_img_path, bbox_inches='tight', dpi=300)
                plt.close(roc_fig)
                
                with open(roc_img_path, "rb") as img_file:
                    roc_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                html_content += f"""
                <div class="chart">
                    <h3>ROC Curves</h3>
                    <img src="data:image/png;base64,{roc_img_base64}" alt="ROC Curves">
                </div>
                """
            
            # Generate PR curve image
            pr_fig = generate_pr_curve_figure(model_data)
            if pr_fig:
                pr_img_path = os.path.join(temp_dir, 'pr_curve.png')
                pr_fig.savefig(pr_img_path, bbox_inches='tight', dpi=300)
                plt.close(pr_fig)
                
                with open(pr_img_path, "rb") as img_file:
                    pr_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                html_content += f"""
                <div class="chart">
                    <h3>Precision-Recall Curves</h3>
                    <img src="data:image/png;base64,{pr_img_base64}" alt="Precision-Recall Curves">
                </div>
                """
        
        # Close HTML document
        html_content += "</body></html>"
        
        # Prepare response
        response = HttpResponse(html_content, content_type='text/html')
        response['Content-Disposition'] = f'attachment; filename="{report_name}.html"'
        return response
        
    except Exception as e:
        return JsonResponse({'error': f'HTML generation failed: {str(e)}'}, status=500)

def generate_csv_report(model_data, report_name, options):
    """
    Generates a CSV report of model metrics.
    
    Args:
        model_data: Dictionary containing model training results
        report_name: Base name for the report file
        options: Dictionary of report options
        
    Returns:
        FileResponse with the CSV report
    """
    try:
        if not options.get('include_metrics', True):
            return JsonResponse({'error': 'CSV reports only include metrics'}, status=400)
        
        # Prepare CSV data
        results = model_data.get('results', [])
        if not results:
            return JsonResponse({'error': 'No results data available'}, status=400)
        
        # Create DataFrame from results
        data = []
        for result in results:
            if 'error' in result:
                row = {
                    'Algorithm': result['algorithm'],
                    'Error': result['error'],
                    'Accuracy': '',
                    'Precision': '',
                    'Recall': '',
                    'F1 Score': ''
                }
            else:
                row = {
                    'Algorithm': result['algorithm'],
                    'Error': '',
                    'Accuracy': result.get('accuracy', ''),
                    'Precision': result.get('precision', ''),
                    'Recall': result.get('recall', ''),
                    'F1 Score': result.get('f1_score', '')
                }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{report_name}.csv"'
        
        df.to_csv(response, index=False)
        return response
        
    except Exception as e:
        return JsonResponse({'error': f'CSV generation failed: {str(e)}'}, status=500)