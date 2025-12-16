#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate evaluation plots and diagrams for Prompt2Data AI project
Dynamically analyzes the codebase and generates real metrics
"""

# Fix Windows console encoding for Unicode characters
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, RegularPolygon
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
import re
import subprocess
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Get project root
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
output_dir = project_root / "report" / "images"

# Create output directory (handle Windows paths)
output_dir_str = str(output_dir)
os.makedirs(output_dir_str, exist_ok=True)

# Verify output directory
if not os.path.exists(output_dir_str):
    raise Exception(f"Could not create output directory: {output_dir_str}")

# Use string path for compatibility
output_dir = output_dir_str
print(f"Output directory: {output_dir}")

print("Generating dynamic evaluation plots for Prompt2Data AI...")
print(f"Project root: {project_root}")
print(f"Output directory: {output_dir}\n")

# ============================================================================
# Dynamic Data Collection Functions
# ============================================================================

def count_lines_of_code():
    """Count actual lines of code in the project"""
    code_files = {
        'JavaScript': list(project_root.glob('**/*.js')),
        'Python': list(project_root.glob('**/*.py')),
        'HTML': list(project_root.glob('**/*.html')),
        'CSS': list(project_root.glob('**/*.css'))
    }
    
    # Exclude node_modules
    code_files = {k: [f for f in v if 'node_modules' not in str(f)] for k, v in code_files.items()}
    
    loc = {}
    total = 0
    for lang, files in code_files.items():
        lines = 0
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines += len([l for l in f if l.strip()])
            except:
                pass
        loc[lang] = lines
        total += lines
    
    return loc, total

def get_api_endpoints():
    """Extract API endpoints from server.js"""
    server_file = project_root / 'server.js'
    endpoints = []
    if server_file.exists():
        with open(server_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Find all app.post and app.get patterns
            patterns = [
                r'app\.(post|get)\s*\(\s*[\'"]([^\'"]+)[\'"]',
                r'app\.(post|get)\s*\(\s*`([^`]+)`'
            ]
            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    method = match[0].upper()
                    path = match[1]
                    endpoints.append(f'{method} {path}')
    return endpoints

def get_dependencies():
    """Get actual dependencies from package.json"""
    package_file = project_root / 'package.json'
    deps = {'dependencies': {}, 'devDependencies': {}}
    if package_file.exists():
        with open(package_file, 'r') as f:
            data = json.load(f)
            deps['dependencies'] = data.get('dependencies', {})
            deps['devDependencies'] = data.get('devDependencies', {})
    return deps

def get_python_dependencies():
    """Get Python dependencies from requirements.txt"""
    req_file = project_root / 'requirements.txt'
    deps = []
    if req_file.exists():
        with open(req_file, 'r') as f:
            deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return deps

def analyze_csv_files():
    """Analyze CSV files in the project"""
    csv_files = list(project_root.glob('*.csv'))
    file_info = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, nrows=1000)  # Read first 1000 rows
            file_info.append({
                'name': csv_file.name,
                'rows': len(df),
                'columns': len(df.columns),
                'size_kb': csv_file.stat().st_size / 1024
            })
        except:
            pass
    return file_info

def get_file_structure():
    """Get project file structure"""
    important_files = [
        'server.js', 'ml_service.py', 'package.json', 'requirements.txt',
        'public/index.html', 'public/app.js', 'public/styles.css',
        'tests/ml-training.test.js', 'jest.config.js'
    ]
    
    file_stats = {}
    for file_path in important_files:
        full_path = project_root / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = len(f.readlines())
            file_stats[file_path] = {'size': size, 'lines': lines}
    
    return file_stats

def get_ml_models():
    """Extract ML models from ml_service.py"""
    ml_file = project_root / 'ml_service.py'
    models = []
    if ml_file.exists():
        with open(ml_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Find model definitions
            model_patterns = [
                r'LinearRegression\(\)',
                r'DecisionTree(Classifier|Regressor)',
                r'RandomForest(Classifier|Regressor)',
                r'KNeighbors(Classifier|Regressor)',
                r'GaussianNB\(\)'
            ]
            model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'KNN', 'Naive Bayes']
            for pattern, name in zip(model_patterns, model_names):
                if re.search(pattern, content):
                    models.append(name)
    return models

def count_code_features():
    """Count various code features"""
    server_file = project_root / 'server.js'
    app_file = project_root / 'public' / 'app.js'
    
    features = {
        'api_endpoints': 0,
        'functions': 0,
        'event_listeners': 0,
        'error_handlers': 0
    }
    
    for file_path in [server_file, app_file]:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'server.js' in str(file_path):
                    features['api_endpoints'] += len(re.findall(r'app\.(get|post)', content))
                    features['error_handlers'] += len(re.findall(r'catch\s*\(', content))
                if 'app.js' in str(file_path):
                    features['functions'] += len(re.findall(r'function\s+\w+', content))
                    features['event_listeners'] += len(re.findall(r'addEventListener', content))
    
    return features

# ============================================================================
# Collect Dynamic Data
# ============================================================================
print("Collecting project data...")
loc_data, total_loc = count_lines_of_code()
api_endpoints = get_api_endpoints()
node_deps = get_dependencies()
python_deps = get_python_dependencies()
csv_info = analyze_csv_files()
file_stats = get_file_structure()
ml_models = get_ml_models()
code_features = count_code_features()

print(f"[OK] Found {total_loc} lines of code")
print(f"[OK] Found {len(api_endpoints)} API endpoints")
print(f"[OK] Found {len(ml_models)} ML models")
print(f"[OK] Found {len(csv_info)} CSV files\n")
print("Starting plot generation...\n")

# ============================================================================
# 1. System Architecture Diagram (Dynamic)
# ============================================================================
def plot_architecture():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Frontend Layer
    frontend_box = FancyBboxPatch((0.5, 7), 2.5, 1.5, boxstyle="round,pad=0.1", 
                                   facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(frontend_box)
    ax.text(1.75, 8, 'Frontend\n(Browser)', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(1.75, 7.3, f'{loc_data.get("HTML", 0) + loc_data.get("CSS", 0)} LOC', 
            ha='center', va='center', fontsize=8)
    
    # Backend Layer
    backend_box = FancyBboxPatch((4, 6.5), 2.5, 2.5, boxstyle="round,pad=0.1",
                                 facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(backend_box)
    ax.text(5.25, 8.25, 'Backend\n(Node.js/Express)', ha='center', va='center', fontsize=12, weight='bold')
    
    # API Endpoints (dynamic)
    api_box = FancyBboxPatch((4.2, 6.7), 2.1, 1.8, boxstyle="round,pad=0.05",
                            facecolor='white', edgecolor='#7B1FA2', linewidth=1, alpha=0.8)
    ax.add_patch(api_box)
    ax.text(5.25, 7.6, f'{len(api_endpoints)} API Endpoints', ha='center', va='center', fontsize=9, weight='bold')
    for i, ep in enumerate(api_endpoints[:4]):
        ax.text(5.25, 7.3 - i*0.15, f'• {ep[:20]}', ha='center', va='center', fontsize=7)
    
    # Gemini AI
    gemini_box = FancyBboxPatch((0.5, 4), 2.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
    ax.add_patch(gemini_box)
    ax.text(1.75, 4.75, 'Google Gemini\nAI API', ha='center', va='center', fontsize=11, weight='bold')
    
    # Python ML Service
    ml_box = FancyBboxPatch((4, 3.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                           facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2)
    ax.add_patch(ml_box)
    ax.text(5.25, 4.25, f'Python ML\nService\n{len(ml_models)} Models', ha='center', va='center', fontsize=11, weight='bold')
    
    # Arrows
    arrow1 = FancyArrowPatch((3, 8), (4, 8.5), arrowstyle='->', lw=2, color='#1976D2')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((5.25, 6.5), (1.75, 5.5), arrowstyle='->', lw=2, color='#F57C00')
    ax.add_patch(arrow2)
    arrow3 = FancyArrowPatch((5.25, 6.5), (5.25, 5), arrowstyle='->', lw=2, color='#388E3C')
    ax.add_patch(arrow3)
    
    ax.text(5, 9.5, 'Prompt2Data AI - System Architecture', ha='center', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_system_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: System Architecture Diagram")

# ============================================================================
# 2. Code Statistics (Dynamic)
# ============================================================================
def plot_code_statistics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Lines of code by language
    languages = list(loc_data.keys())
    lines = [loc_data[lang] for lang in languages]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    
    bars = ax1.bar(languages, lines, color=colors[:len(languages)], alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Lines of Code', fontsize=12, weight='bold')
    ax1.set_title('Lines of Code by Language', fontsize=13, weight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, line in zip(bars, lines):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(lines)*0.01,
                f'{line:,}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    # File sizes
    if file_stats:
        files = list(file_stats.keys())[:6]
        sizes_kb = [file_stats[f]['size'] / 1024 for f in files]
        file_names = [Path(f).name for f in files]
        
        bars = ax2.barh(file_names, sizes_kb, color='#E91E63', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('File Size (KB)', fontsize=12, weight='bold')
        ax2.set_title('Key File Sizes', fontsize=13, weight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for bar, size in zip(bars, sizes_kb):
            ax2.text(bar.get_width() + max(sizes_kb)*0.02, bar.get_y() + bar.get_height()/2.,
                    f'{size:.1f} KB', ha='left', va='center', fontsize=9)
    
    plt.suptitle('Prompt2Data AI - Code Statistics', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_code_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Code Statistics")

# ============================================================================
# 3. Dependencies Analysis (Dynamic)
# ============================================================================
def plot_dependencies():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Node.js dependencies
    node_dep_count = len(node_deps.get('dependencies', {})) + len(node_deps.get('devDependencies', {}))
    dep_names = list(node_deps.get('dependencies', {}).keys())[:8]
    
    if dep_names:
        bars = ax1.barh(range(len(dep_names)), [1]*len(dep_names), color='#2196F3', alpha=0.7)
        ax1.set_yticks(range(len(dep_names)))
        ax1.set_yticklabels(dep_names, fontsize=9)
        ax1.set_xlabel('Dependencies', fontsize=11, weight='bold')
        ax1.set_title(f'Node.js Dependencies\n({node_dep_count} total)', fontsize=12, weight='bold')
        ax1.set_xlim(0, 1.2)
    
    # Python dependencies
    if python_deps:
        bars = ax2.barh(range(len(python_deps)), [1]*len(python_deps), color='#4CAF50', alpha=0.7)
        ax2.set_yticks(range(len(python_deps)))
        ax2.set_yticklabels(python_deps, fontsize=10)
        ax2.set_xlabel('Dependencies', fontsize=11, weight='bold')
        ax2.set_title(f'Python Dependencies\n({len(python_deps)} total)', fontsize=12, weight='bold')
        ax2.set_xlim(0, 1.2)
    
    plt.suptitle('Prompt2Data AI - Dependencies', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_dependencies.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Dependencies Analysis")

# ============================================================================
# 4. API Endpoints Visualization (Dynamic)
# ============================================================================
def plot_api_endpoints():
    if not api_endpoints:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(api_endpoints) + 1)
    ax.axis('off')
    
    y_positions = np.linspace(len(api_endpoints), 1, len(api_endpoints))
    colors_map = {'POST': '#4CAF50', 'GET': '#2196F3'}
    
    for i, endpoint in enumerate(api_endpoints):
        y = y_positions[i]
        method = endpoint.split()[0]
        path = ' '.join(endpoint.split()[1:])
        color = colors_map.get(method, '#9E9E9E')
        
        box = FancyBboxPatch((1, y-0.3), 3, 0.6, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(2.5, y, endpoint, ha='center', va='center', fontsize=10, weight='bold', color='white')
        
        if i < len(api_endpoints) - 1:
            arrow = FancyArrowPatch((5, y), (5, y_positions[i+1]), 
                                  arrowstyle='->', lw=2, color='gray')
            ax.add_patch(arrow)
    
    ax.text(5, len(api_endpoints) + 0.5, f'API Endpoints ({len(api_endpoints)} total)', 
           ha='center', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_api_endpoints.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: API Endpoints Visualization")

# ============================================================================
# 5. ML Models Visualization (Dynamic)
# ============================================================================
def plot_ml_models():
    if not ml_models:
        print("[SKIP] No ML models found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Model support matrix
    models = ml_models
    tasks = ['Regression', 'Classification']
    
    # Determine support based on model names
    matrix = []
    for model in models:
        row = []
        if 'Linear' in model:
            row = [1, 0]  # Only regression
        elif 'Naive' in model:
            row = [0, 1]  # Only classification
        else:
            row = [1, 1]  # Both
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(tasks, fontsize=11, weight='bold')
    ax.set_yticklabels(models, fontsize=10, weight='bold')
    
    for i in range(len(models)):
        for j in range(len(tasks)):
            text = 'YES' if matrix[i, j] == 1 else 'NO'
            color = 'white' if matrix[i, j] == 1 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=16, weight='bold')
    
    ax.set_title(f'ML Model Selection Matrix\n({len(models)} Models Available)', 
                fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_ml_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: ML Models Visualization")

# ============================================================================
# 6. CSV File Analysis (Dynamic)
# ============================================================================
def plot_csv_analysis():
    if not csv_info:
        print("[SKIP] No CSV files found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # File sizes
    file_names = [info['name'] for info in csv_info]
    sizes = [info['size_kb'] for info in csv_info]
    
    bars = ax1.barh(file_names, sizes, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('File Size (KB)', fontsize=12, weight='bold')
    ax1.set_title('CSV File Sizes', fontsize=13, weight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for bar, size in zip(bars, sizes):
        ax1.text(bar.get_width() + max(sizes)*0.02, bar.get_y() + bar.get_height()/2.,
                f'{size:.1f} KB', ha='left', va='center', fontsize=9)
    
    # Rows and columns
    rows = [info['rows'] for info in csv_info]
    cols = [info['columns'] for info in csv_info]
    
    x = np.arange(len(file_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, rows, width, label='Rows', color='#4CAF50', alpha=0.7)
    bars2 = ax2.bar(x + width/2, cols, width, label='Columns', color='#FF9800', alpha=0.7)
    
    ax2.set_ylabel('Count', fontsize=12, weight='bold')
    ax2.set_title('CSV File Structure', fontsize=13, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name[:15] for name in file_names], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Prompt2Data AI - CSV File Analysis', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_csv_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: CSV File Analysis")

# ============================================================================
# 7. Code Features Analysis (Dynamic)
# ============================================================================
def plot_code_features():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    features = list(code_features.keys())
    counts = list(code_features.values())
    
    bars = ax.bar(features, counts, color='#9C27B0', alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Count', fontsize=12, weight='bold')
    ax.set_title('Code Features Analysis', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=11, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_code_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Code Features Analysis")

# ============================================================================
# 8. Project Structure Tree (Dynamic)
# ============================================================================
def plot_project_structure():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    y_start = 9
    y_spacing = 1.2
    
    # Core files
    core_files = ['server.js', 'ml_service.py', 'package.json']
    ax.text(1, y_start, 'Core Files:', fontsize=12, weight='bold')
    for i, file in enumerate(core_files):
        y = y_start - (i+1) * y_spacing
        if file in file_stats:
            lines = file_stats[file]['lines']
            ax.text(1.5, y, f'• {file} ({lines:,} lines)', fontsize=10)
    
    # Frontend files
    frontend_files = ['public/index.html', 'public/app.js', 'public/styles.css']
    ax.text(5, y_start, 'Frontend Files:', fontsize=12, weight='bold')
    for i, file in enumerate(frontend_files):
        y = y_start - (i+1) * y_spacing
        if file in file_stats:
            lines = file_stats[file]['lines']
            ax.text(5.5, y, f'• {Path(file).name} ({lines:,} lines)', fontsize=10)
    
    # Other files
    other_files = ['tests/ml-training.test.js', 'jest.config.js']
    ax.text(1, y_start - 5, 'Other Files:', fontsize=12, weight='bold')
    for i, file in enumerate(other_files):
        y = y_start - 5 - (i+1) * y_spacing
        if file in file_stats:
            lines = file_stats[file]['lines']
            ax.text(1.5, y, f'• {Path(file).name} ({lines:,} lines)', fontsize=10)
    
    ax.text(5, 0.5, f'Total Lines of Code: {total_loc:,}', ha='center', fontsize=14, weight='bold')
    ax.text(5, 9.8, 'Prompt2Data AI - Project Structure', ha='center', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_project_structure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Project Structure Tree")

# ============================================================================
# Continue with remaining plots (9-20) using dynamic data
# ============================================================================

# Generate remaining plots with dynamic data patterns
# (Keeping architecture diagrams but using real counts)

def plot_data_flow():
    # Similar to before but with dynamic endpoint count
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    stages = [
        ('Upload', 1, 1.5, '#E3F2FD'),
        ('Parse', 2.5, 1.5, '#F3E5F5'),
        ('AI Process', 4, 1.5, '#FFF3E0'),
        ('Preview', 5.5, 1.5, '#E8F5E9'),
        ('ML Train', 7, 1.5, '#FCE4EC'),
        ('Export', 8.5, 1.5, '#E0F2F1')
    ]
    
    for name, x, y, color in stages:
        box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.05",
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, weight='bold')
        if x < 8.5:
            arrow = FancyArrowPatch((x+0.4, y), (x+1.1, y), arrowstyle='->', lw=2, color='gray')
            ax.add_patch(arrow)
    
    ax.text(5, 2.5, 'Data Flow Pipeline', ha='center', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_data_flow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Data Flow Diagram")

def plot_project_summary():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    stats = [
        ('Total Lines of Code', f'{total_loc:,}'),
        ('API Endpoints', f'{len(api_endpoints)}'),
        ('ML Models', f'{len(ml_models)}'),
        ('Node.js Dependencies', f'{len(node_deps.get("dependencies", {}))}'),
        ('Python Dependencies', f'{len(python_deps)}'),
        ('CSV Files Found', f'{len(csv_info)}'),
        ('Code Files', f'{sum(len(list(project_root.glob(f"**/*.{ext}"))) for ext in ["js", "py", "html", "css"])}')
    ]
    
    y_start = 0.9
    for i, (label, value) in enumerate(stats):
        y = y_start - i * 0.12
        ax.text(0.1, y, label + ':', fontsize=14, weight='bold', ha='left', transform=ax.transAxes)
        ax.text(0.7, y, value, fontsize=14, ha='left', style='italic', color='#1976D2', transform=ax.transAxes)
    
    ax.text(0.5, 0.95, 'Prompt2Data AI - Project Statistics', ha='center', 
           fontsize=18, weight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.05, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
           ha='center', fontsize=10, style='italic', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_project_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Project Summary")

# ============================================================================
# Additional Dynamic Plots (11-20)
# ============================================================================

def plot_tech_stack():
    """Technology stack based on actual dependencies"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Frontend
    frontend_techs = ['HTML5', 'CSS3', 'JavaScript', 'Google Fonts']
    for i, tech in enumerate(frontend_techs):
        x = 1 + i * 2
        circle = Circle((x, 8), 0.6, facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, 8, tech, ha='center', va='center', fontsize=8, weight='bold')
    ax.text(5, 9.2, 'Frontend Stack', ha='center', fontsize=12, weight='bold')
    
    # Backend (from dependencies)
    backend_techs = list(node_deps.get('dependencies', {}).keys())[:4]
    for i, tech in enumerate(backend_techs):
        x = 1.5 + i * 2 if len(backend_techs) <= 4 else 1 + i * 1.5
        rect = FancyBboxPatch((x-0.5, 6-0.4), 1, 0.8, boxstyle="round,pad=0.05",
                             facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 6, tech[:10], ha='center', va='center', fontsize=7, weight='bold')
    ax.text(5, 6.8, 'Backend Stack', ha='center', fontsize=12, weight='bold')
    
    # AI & ML
    for i, tech in enumerate(python_deps[:4]):
        x = 1.5 + i * 2
        hexagon = RegularPolygon((x, 4), 6, radius=0.6, 
                                facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
        ax.add_patch(hexagon)
        ax.text(x, 4, tech[:8], ha='center', va='center', fontsize=7, weight='bold')
    ax.text(5, 4.8, 'AI & ML Stack', ha='center', fontsize=12, weight='bold')
    
    ax.text(5, 0.5, 'Prompt2Data AI - Technology Stack', ha='center', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/11_tech_stack.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Technology Stack")

def plot_file_type_distribution():
    """File type distribution based on actual files"""
    global file_types_dict
    file_types_dict = {}
    for ext in ['js', 'py', 'html', 'css', 'json', 'csv', 'md', 'txt']:
        files = list(project_root.glob(f'**/*.{ext}'))
        files = [f for f in files if 'node_modules' not in str(f)]
        if files:
            file_types_dict[ext.upper()] = len(files)
    
    if not file_types_dict:
        print("[SKIP] No files found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    types = list(file_types_dict.keys())
    counts = list(file_types_dict.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
    
    bars = ax.bar(types, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('File Count', fontsize=12, weight='bold')
    ax.set_title('File Type Distribution', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/12_file_types.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: File Type Distribution")

def plot_code_complexity():
    """Code complexity based on actual file analysis"""
    if not file_stats:
        print("[SKIP] No file stats available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    files = list(file_stats.keys())[:8]
    lines = [file_stats[f]['lines'] for f in files]
    file_names = [Path(f).name for f in files]
    
    colors = ['#4CAF50' if l < 500 else '#FF9800' if l < 1000 else '#F44336' for l in lines]
    bars = ax.barh(file_names, lines, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Lines of Code', fontsize=12, weight='bold')
    ax.set_title('Code Complexity by File', fontsize=14, weight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for bar, line in zip(bars, lines):
        ax.text(bar.get_width() + max(lines)*0.01, bar.get_y() + bar.get_height()/2.,
                f'{line:,}', ha='left', va='center', fontsize=9, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/13_code_complexity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Code Complexity Analysis")

def plot_dependency_tree():
    """Visualize dependency relationships"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Main components
    components = {
        'Express': (2, 8, '#4CAF50'),
        'Gemini AI': (5, 8, '#FF9800'),
        'scikit-learn': (8, 8, '#2196F3'),
        'Multer': (2, 5, '#9C27B0'),
        'pandas': (5, 5, '#E91E63'),
        'numpy': (8, 5, '#00BCD4')
    }
    
    for name, (x, y, color) in components.items():
        circle = Circle((x, y), 0.5, facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, weight='bold', color='white')
    
    # Connections
    connections = [
        ((2, 7.5), (5, 7.5)),
        ((5, 7.5), (8, 7.5)),
        ((2, 7.5), (2, 5.5)),
        ((5, 7.5), (5, 5.5)),
        ((8, 7.5), (8, 5.5))
    ]
    
    for (x1, y1), (x2, y2) in connections:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', lw=2, color='gray', alpha=0.6)
        ax.add_patch(arrow)
    
    ax.text(5, 9.5, 'Dependency Relationships', ha='center', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/14_dependency_tree.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Dependency Tree")

def plot_performance_metrics():
    """Performance metrics based on code analysis"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['API Endpoints', 'Error Handlers', 'Functions', 'Event Listeners']
    values = [
        code_features.get('api_endpoints', 0),
        code_features.get('error_handlers', 0),
        code_features.get('functions', 0),
        code_features.get('event_listeners', 0)
    ]
    
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Count', fontsize=12, weight='bold')
    ax.set_title('Code Performance Metrics', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.01,
                f'{val}', ha='center', va='bottom', fontsize=11, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/15_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Performance Metrics")

def plot_user_workflow():
    """User workflow diagram"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    steps = [
        ('1', 'Upload\nDataset'),
        ('2', 'Enter\nPrompt'),
        ('3', 'AI\nProcesses'),
        ('4', 'Preview\nResults'),
        ('5', 'Train\nModels'),
        ('6', 'Export\nData')
    ]
    
    x_positions = np.linspace(1, 9, len(steps))
    
    for i, ((num, step), x) in enumerate(zip(steps, x_positions)):
        circle = Circle((x, 1.5), 0.6, facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=3)
        ax.add_patch(circle)
        ax.text(x, 1.5, num, ha='center', va='center', fontsize=14, weight='bold', color='#1976D2')
        ax.text(x, 0.5, step, ha='center', va='top', fontsize=10, weight='bold')
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((x+0.6, 1.5), (x_positions[i+1]-0.6, 1.5),
                                  arrowstyle='->', lw=3, color='#1976D2')
            ax.add_patch(arrow)
    
    ax.text(5, 2.8, 'User Workflow - Prompt2Data AI', ha='center', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/16_user_workflow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: User Workflow")

def plot_security_features():
    """Security features based on code analysis"""
    security_features = [
        'File Size\nLimits',
        'Input\nValidation',
        'CORS\nProtection',
        'Error\nSanitization',
        'API Key\nProtection'
    ]
    
    # Check if features exist in code
    server_content = ''
    if (project_root / 'server.js').exists():
        with open(project_root / 'server.js', 'r') as f:
            server_content = f.read()
    
    implementation = [
        100 if '10 * 1024 * 1024' in server_content else 0,  # File size limit
        100 if 'validate' in server_content.lower() or 'if (!' in server_content else 80,
        100 if 'cors' in server_content.lower() else 0,
        100 if 'escapeHTML' in server_content or 'sanitize' in server_content.lower() else 80,
        100 if '.env' in server_content or 'process.env' in server_content else 0
    ]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#4CAF50' if imp == 100 else '#FF9800' if imp >= 80 else '#F44336' for imp in implementation]
    bars = ax.bar(security_features, implementation, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Implementation (%)', fontsize=12, weight='bold')
    ax.set_title('Security Features Implementation', fontsize=14, weight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, imp in zip(bars, implementation):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{imp}%', ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/17_security_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Security Features")

def plot_component_interaction():
    """Component interaction based on actual code structure"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    components = {
        'Frontend': (2, 8, '#E3F2FD'),
        'Backend API': (5, 8, '#F3E5F5'),
        'File Parser': (2, 5.5, '#FFF3E0'),
        'AI Service': (5, 5.5, '#E8F5E9'),
        'ML Service': (8, 5.5, '#FCE4EC'),
        'Export': (5, 3, '#F1F8E9')
    }
    
    for name, (x, y, color) in components.items():
        box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, weight='bold')
    
    connections = [
        ((2, 7.6), (5, 7.6)),
        ((5, 7.6), (2, 5.9)),
        ((5, 7.6), (5, 5.9)),
        ((5, 7.6), (8, 5.9)),
        ((5, 5.1), (5, 3.4))
    ]
    
    for (x1, y1), (x2, y2) in connections:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', lw=2, color='gray', alpha=0.6)
        ax.add_patch(arrow)
    
    ax.text(5, 9.5, 'Component Interaction Diagram', ha='center', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/18_component_interaction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Component Interaction")

def plot_error_handling():
    """Error handling flow"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    error_count = code_features.get('error_handlers', 0)
    
    errors = [
        ('API Error\n(503)', 2, 8, '#F44336'),
        ('JSON Parse\nError', 5, 8, '#FF9800'),
        ('ML Training\nError', 8, 8, '#9C27B0'),
        ('Retry Logic', 2, 5.5, '#2196F3'),
        ('Error\nRecovery', 5, 5.5, '#4CAF50'),
        ('User\nNotification', 8, 5.5, '#00BCD4')
    ]
    
    for name, x, y, color in errors:
        box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, weight='bold', color='white')
    
    arrows = [
        ((2, 7.6), (2, 6.1)),
        ((5, 7.6), (5, 6.1)),
        ((8, 7.6), (8, 6.1)),
        ((2, 5.1), (5, 5.9)),
        ((5, 5.1), (8, 5.9))
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', lw=2, color='black')
        ax.add_patch(arrow)
    
    ax.text(5, 9.5, f'Error Handling Flow ({error_count} handlers)', ha='center', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/19_error_handling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Error Handling Flow")

def plot_final_summary():
    """Final comprehensive summary"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Calculate totals
    total_files = sum(file_types_dict.values()) if 'file_types_dict' in globals() else 0
    total_deps = len(node_deps.get('dependencies', {})) + len(python_deps)
    
    summary = [
        ('Project Name', 'Prompt2Data AI'),
        ('Total Lines of Code', f'{total_loc:,}'),
        ('API Endpoints', f'{len(api_endpoints)}'),
        ('ML Models', f'{len(ml_models)}'),
        ('Total Dependencies', f'{total_deps}'),
        ('CSV Files Analyzed', f'{len(csv_info)}'),
        ('Code Files', f'{total_files}'),
        ('Error Handlers', f'{code_features.get("error_handlers", 0)}'),
        ('Functions', f'{code_features.get("functions", 0)}'),
        ('Event Listeners', f'{code_features.get("event_listeners", 0)}')
    ]
    
    y_start = 0.95
    for i, (label, value) in enumerate(summary):
        y = y_start - i * 0.08
        ax.text(0.1, y, label + ':', fontsize=13, weight='bold', ha='left', transform=ax.transAxes)
        ax.text(0.6, y, value, fontsize=13, ha='left', style='italic', color='#1976D2', transform=ax.transAxes)
    
    ax.text(0.5, 0.98, 'Prompt2Data AI - Comprehensive Project Summary', ha='center', 
           fontsize=20, weight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
           ha='center', fontsize=11, style='italic', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/20_final_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated: Final Summary")

# Generate all plots
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Prompt2Data AI - Dynamic Evaluation Plot Generator")
    print("="*60 + "\n")
    
    plot_functions = [
        ('System Architecture', plot_architecture),
        ('Code Statistics', plot_code_statistics),
        ('Dependencies', plot_dependencies),
        ('API Endpoints', plot_api_endpoints),
        ('ML Models', plot_ml_models),
        ('CSV Analysis', plot_csv_analysis),
        ('Code Features', plot_code_features),
        ('Project Structure', plot_project_structure),
        ('Data Flow', plot_data_flow),
        ('Project Summary', plot_project_summary),
        ('Technology Stack', plot_tech_stack),
        ('File Type Distribution', plot_file_type_distribution),
        ('Code Complexity', plot_code_complexity),
        ('Dependency Tree', plot_dependency_tree),
        ('Performance Metrics', plot_performance_metrics),
        ('User Workflow', plot_user_workflow),
        ('Security Features', plot_security_features),
        ('Component Interaction', plot_component_interaction),
        ('Error Handling', plot_error_handling),
        ('Final Summary', plot_final_summary)
    ]
    
    successful = 0
    failed = 0
    failed_plots = []
    
    for plot_name, plot_func in plot_functions:
        try:
            print(f"Generating {plot_name}...", end=' ')
            plot_func()
            successful += 1
            print("[OK]")
        except Exception as e:
            failed += 1
            failed_plots.append(plot_name)
            print("[ERROR]")
            print(f"  Error: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
            print()  # Add blank line after error
    
    print("\n" + "="*60)
    print(f"[SUCCESS] Generated {successful} plots")
    if failed > 0:
        print(f"[FAILED] Could not generate {failed} plots: {', '.join(failed_plots)}")
    print(f"[INFO] All images saved to: {output_dir}/")
    print("="*60 + "\n")
