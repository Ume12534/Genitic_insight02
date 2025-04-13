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
    Handle sequence feature extraction request and return CSV data without ID column.
    Updated to handle temporary file storage and cleanup with proper file handling.
    
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
    if request.method == 'POST' and request.FILES.get('fasta_file'):
        # Initialize file paths for cleanup
        input_file_path = None
        output_file_path = None
        
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
            # Create temp directories if they don't exist
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            temp_input_dir = os.path.join(temp_dir, 'input')
            temp_output_dir = os.path.join(temp_dir, 'output')
            
            os.makedirs(temp_input_dir, exist_ok=True)
            os.makedirs(temp_output_dir, exist_ok=True)
            
            # Clean up any previous temporary files
            for f in os.listdir(temp_input_dir):
                try:
                    os.remove(os.path.join(temp_input_dir, f))
                except PermissionError:
                    continue  # Skip files that can't be deleted
            for f in os.listdir(temp_output_dir):
                try:
                    os.remove(os.path.join(temp_output_dir, f))
                except PermissionError:
                    continue
            
            # =============================================
            # 3. SAVE UPLOADED FILE TO TEMP LOCATION
            # =============================================
            # Generate unique filename to prevent collisions
            import uuid
            unique_id = uuid.uuid4().hex
            input_file_path = os.path.join(temp_input_dir, f"{unique_id}_{fasta_file.name}")
            
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
            output_file_path = os.path.join(temp_output_dir, f"{unique_id}_features.csv")
            
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
                
                # Save processed CSV to temp location
                try:
                    with open(output_file_path, 'w') as f:
                        f.write(csv_data)
                except IOError as e:
                    return JsonResponse({
                        'error': 'Failed to save results',
                        'details': str(e)
                    }, status=500)
            


            # =============================================
            # 7. STORE EXTRACTED DATA FOR FUTURE USE
            # =============================================
            if csv_data:
                # Generate unique extraction ID
                extraction_id = str(uuid.uuid4())
                
                # Create temp file path for the extracted features
                features_file_path = os.path.join(temp_output_dir, f"{extraction_id}_features.csv")
                
                # Save the CSV data to file
                with open(features_file_path, 'w') as f:
                    f.write(csv_data)
                
                # Store metadata in global dictionary
                EXTRACTED_DATA_STORE[extraction_id] = {
                    'file_path': features_file_path,
                    'sequence_type': sequence_type,
                    'descriptor': descriptor,
                    'created_at': time.time(),
                    'target_column': None  # Will be determined during training
                }
                
                # Also store in session for immediate access
                request.session['extracted_features'] = {
                    'extraction_id': extraction_id,
                    'csv_data': csv_data  # Store a copy of the data
                }


            # =============================================
            # 8. PREPARE RESPONSE
            # =============================================
            response_data = {
                'message': 'Feature extraction completed',
                'file_name': fasta_file.name,
                'sequence_type': sequence_type,
                'descriptor': descriptor,
                'csv_data': csv_data,  # This now has no ID column
                'temp_file_path': output_file_path  # Store path for potential download
            }
            
            # =============================================
            # 9. CLEAN UP INPUT FILE (KEEP OUTPUT FOR NOW)
            # =============================================
            try:
                if input_file_path and os.path.exists(input_file_path):
                    os.remove(input_file_path)
            except Exception as e:
                print(f"Warning: Could not delete input file: {e}")
            
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
            
            try:
                if output_file_path and os.path.exists(output_file_path):
                    os.remove(output_file_path)
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
                    # Get extraction_id from session
                    extraction_id = request.session.get('extracted_features', {}).get('extraction_id')
                   
                    if not extraction_id:
                        return JsonResponse({'error': 'No extraction ID found in session'}, status=400)

                    # Construct the file path from extraction ID
                    temp_output_dir = os.path.join(settings.MEDIA_ROOT, 'temp', 'output')
                    features_file_path = os.path.join(temp_output_dir, f"{extraction_id}_features.csv")

                    # Check if file exists
                    if not os.path.exists(features_file_path):
                        return JsonResponse({'error': f'No features file found with ID {extraction_id}'}, status=400)
                    
                    # Read CSV into DataFrame
                    df = pd.read_csv(features_file_path)
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