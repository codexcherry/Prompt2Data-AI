#!/usr/bin/env python3
"""
ML Service for Dataset Preprocessor
Trains machine learning models and returns evaluation metrics.
Communicates via stdin/stdout JSON.
"""

import sys
import json
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

warnings.filterwarnings('ignore')


def detect_task_type(target_values):
    """
    Auto-detect if task is classification or regression.
    Properly handles numeric values stored as strings.
    """
    if not target_values or len(target_values) == 0:
        return 'regression'  # Default
    
    # Try to convert all values to numeric first
    numeric_values = []
    non_numeric_count = 0
    
    for val in target_values:
        if val is None or val == '' or pd.isna(val):
            continue
        try:
            # Try to convert to float
            num_val = float(val)
            numeric_values.append(num_val)
        except (ValueError, TypeError):
            # Can't convert to number - it's categorical
            non_numeric_count += 1
    
    # If we have any non-numeric values, it's classification
    if non_numeric_count > 0:
        return 'classification'
    
    # If no numeric values found, it's classification
    if len(numeric_values) == 0:
        return 'classification'
    
    # Check if it's classification based on unique values
    unique_count = len(set(numeric_values))
    total_count = len(numeric_values)
    
    # If very few unique values relative to total, likely classification
    # But if we have many unique numeric values, it's regression
    if unique_count <= 5 and total_count > 10:
        return 'classification'
    
    # Default to regression for numeric values
    return 'regression'


def preprocess_data(df, target_column):
    """
    Preprocess data for ML training.
    Properly converts string numbers to numeric types.
    """
    # Separate features and target
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    
    # Convert numeric columns that might be strings
    for col in X.columns:
        if X[col].dtype == 'object':
            # Try to convert to numeric
            try:
                X[col] = pd.to_numeric(X[col], errors='ignore')
            except:
                pass
    
    # Handle target: try to convert to numeric if possible
    try:
        y_numeric = pd.to_numeric(y, errors='coerce')
        numeric_count = y_numeric.notna().sum()
        # If 80%+ are numeric, use numeric version
        if numeric_count > len(y) * 0.8:
            y = y_numeric
    except:
        pass
    
    # Encode categorical features (non-numeric)
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Encode target if still categorical
    target_encoder = None
    if y.dtype == 'object' or str(y.dtype) == 'object':
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y.astype(str))
        y = pd.Series(y_encoded)
    else:
        # Ensure numeric target is float
        y = pd.to_numeric(y, errors='coerce')
    
    # Handle missing values in features
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)
        else:
            X[col] = X[col].fillna(0)
    
    # Handle missing values in target
    if y.isna().any():
        if y.dtype in ['float64', 'int64', 'float32', 'int32']:
            median_val = y.median()
            if pd.isna(median_val):
                y = y.fillna(0)
            else:
                y = y.fillna(median_val)
        else:
            mode_vals = y.mode()
            fill_val = mode_vals[0] if len(mode_vals) > 0 else 0
            y = y.fillna(fill_val)
    
    # Convert to numeric arrays - ensure all features are numeric
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.empty:
        # If no numeric columns, use encoded categorical
        X_numeric = X
    
    # Ensure we have at least one feature
    if X_numeric.shape[1] == 0:
        raise ValueError("No valid features after preprocessing")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Convert y to numpy array
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = np.array(y)
    
    # Determine appropriate dtype
    if y_values.dtype in [np.float64, np.float32, np.int64, np.int32]:
        y_array = y_values.astype(float)
    else:
        y_array = y_values.astype(int)
    
    return X_scaled, y_array, label_encoders, target_encoder


def get_classification_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Handle NaN values - replace with None (which becomes null in JSON)
    return {
        'accuracy': float(accuracy) if not np.isnan(accuracy) else None,
        'precision': float(precision) if not np.isnan(precision) else None,
        'recall': float(recall) if not np.isnan(recall) else None,
        'f1_score': float(f1) if not np.isnan(f1) else None
    }


def get_regression_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Handle NaN values - replace with None (which becomes null in JSON)
    r2_score_value = float(r2) if not np.isnan(r2) else None
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2_score_value
    }


def train_model(model_id, X_train, X_test, y_train, y_test, task_type):
    """Train a single model and return results."""
    start_time = time.time()
    
    try:
        # Select model based on ID and task type
        if model_id == 'linear_regression':
            if task_type == 'classification':
                return {'success': False, 'error': 'Linear Regression not suitable for classification'}
            model = LinearRegression()
        
        elif model_id == 'decision_tree':
            if task_type == 'classification':
                model = DecisionTreeClassifier(random_state=42)
            else:
                model = DecisionTreeRegressor(random_state=42)
        
        elif model_id == 'random_forest':
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        elif model_id == 'knn':
            n_neighbors = min(5, len(X_train))
            if task_type == 'classification':
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
            else:
                model = KNeighborsRegressor(n_neighbors=n_neighbors)
        
        elif model_id == 'naive_bayes':
            if task_type == 'regression':
                return {'success': False, 'error': 'Naive Bayes not suitable for regression'}
            model = GaussianNB()
        
        else:
            return {'success': False, 'error': f'Unknown model: {model_id}'}
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if task_type == 'classification':
            metrics = get_classification_metrics(y_test, y_pred)
        else:
            metrics = get_regression_metrics(y_test, y_pred)
        
        training_time = int((time.time() - start_time) * 1000)
        
        return {
            'success': True,
            'metrics': metrics,
            'trainingTime': training_time
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'trainingTime': int((time.time() - start_time) * 1000)
        }


def find_best_model(results, task_type):
    """Find the best performing model based on primary metric."""
    best_model = None
    best_score = None
    
    primary_metric = 'accuracy' if task_type == 'classification' else 'r2_score'
    
    for model_id, result in results.items():
        if result.get('success') and 'metrics' in result:
            score = result['metrics'].get(primary_metric, 0)
            if best_score is None or score > best_score:
                best_score = score
                best_model = model_id
    
    return best_model


def train_models(data, target_column, models, train_test_split_ratio, task_type=None):
    """
    Train multiple ML models and return metrics.
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if target_column not in df.columns:
            return {'success': False, 'error': f'Target column "{target_column}" not found'}
        
        # Auto-detect task type if not provided
        if task_type is None:
            task_type = detect_task_type(df[target_column].tolist())
        
        # Preprocess data
        X, y, _, _ = preprocess_data(df, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1 - train_test_split_ratio), random_state=42
        )
        
        # Train each model
        results = {}
        for model_id in models:
            results[model_id] = train_model(
                model_id, X_train, X_test, y_train, y_test, task_type
            )
        
        # Find best model
        best_model = find_best_model(results, task_type)
        
        return {
            'success': True,
            'results': results,
            'bestModel': best_model,
            'taskType': task_type
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}


def main():
    """Main entry point - reads JSON from stdin, outputs JSON to stdout."""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        request = json.loads(input_data)
        
        # Extract parameters
        data = request.get('data', [])
        target_column = request.get('targetColumn', '')
        models = request.get('models', [])
        train_test_split_ratio = request.get('trainTestSplit', 0.8)
        task_type = request.get('taskType')
        
        # Train models
        result = train_models(data, target_column, models, train_test_split_ratio, task_type)
        
        # Output result - handle NaN values
        def handle_nan(obj):
            """Recursively replace NaN with None"""
            if isinstance(obj, dict):
                return {k: handle_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [handle_nan(item) for item in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            return obj
        
        result_cleaned = handle_nan(result)
        print(json.dumps(result_cleaned))
    
    except json.JSONDecodeError as e:
        print(json.dumps({'success': False, 'error': f'Invalid JSON input: {str(e)}'}))
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}))


if __name__ == '__main__':
    main()
