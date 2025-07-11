#!/usr/bin/env python3
"""
Setup script for Credit Card Fraud Detection project.

This script:
1. Creates necessary directories
2. Downloads the credit card dataset (if Kaggle API is configured)
3. Sets up the environment
4. Validates the installation

Run this after installing requirements.txt
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def create_directories():
    """Create all necessary project directories."""
    directories = [
        'data/raw',
        'data/processed', 
        'data/sample',
        'models/fraud_detector',
        'models/clustering',
        'reports/figures',
        'reports/analysis',
        'reports/logs',
        'notebooks',
        'tests',
        'docs'
    ]
    
    print("Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")


def check_kaggle_setup():
    """Check if Kaggle API is properly configured."""
    try:
        import kaggle
        return True
    except (ImportError, OSError):
        return False


def download_dataset():
    """Download the credit card fraud dataset from Kaggle."""
    dataset_path = "data/raw/creditcard.csv"
    
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return True
    
    if not check_kaggle_setup():
        print("⚠️  Kaggle API not configured. Please:")
        print("   1. Install kaggle: pip install kaggle")
        print("   2. Set up API credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        print("   3. Or manually download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("   4. Place creditcard.csv in data/raw/ directory")
        return False
    
    try:
        print("Downloading dataset from Kaggle...")
        subprocess.run([
            "kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud",
            "-p", "data/raw", "--unzip"
        ], check=True)
        print("✓ Dataset downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download dataset: {e}")
        return False


def create_sample_data():
    """Create a small sample dataset for testing."""
    source_path = "data/raw/creditcard.csv"
    sample_path = "data/sample/creditcard_sample.csv"
    
    if not os.path.exists(source_path):
        print("⚠️  Main dataset not found, skipping sample creation")
        return
    
    if os.path.exists(sample_path):
        print(f"Sample dataset already exists at {sample_path}")
        return
    
    try:
        import pandas as pd
        print("Creating sample dataset for testing...")
        
        # Read full dataset
        df = pd.read_csv(source_path)
        
        # Create stratified sample (1000 normal + all fraud cases)
        fraud_cases = df[df['Class'] == 1]
        normal_cases = df[df['Class'] == 0].sample(n=1000, random_state=42)
        
        sample_df = pd.concat([normal_cases, fraud_cases]).sample(frac=1, random_state=42)
        sample_df.to_csv(sample_path, index=False)
        
        print(f"✓ Sample dataset created: {len(sample_df)} records")
        print(f"  - Normal transactions: {len(sample_df[sample_df['Class'] == 0])}")
        print(f"  - Fraudulent transactions: {len(sample_df[sample_df['Class'] == 1])}")
        
    except ImportError:
        print("❌ pandas not installed, skipping sample creation")
    except Exception as e:
        print(f"❌ Failed to create sample dataset: {e}")


def validate_installation():
    """Validate that all required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'plotly', 'yaml', 'joblib'
    ]
    
    print("Validating package installation...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    print("✓ All required packages are installed")
    return True


def create_gitignore():
    """Create a comprehensive .gitignore file."""
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files (large datasets)
data/raw/*.csv
data/processed/*.csv
*.parquet

# Model files
models/**/*.joblib
models/**/*.pkl
models/**/*.h5

# Logs
reports/logs/*.log

# Jupyter Notebook checkpoints
.ipynb_checkpoints

# Temporary files
*.tmp
*.temp

# Configuration (if sensitive)
config/secrets.yaml
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✓ .gitignore created")


def main():
    """Main setup function."""
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION - PROJECT SETUP")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Create .gitignore
    create_gitignore()
    
    # Validate installation
    if not validate_installation():
        sys.exit(1)
    
    # Download dataset
    dataset_downloaded = download_dataset()
    
    # Create sample data
    if dataset_downloaded:
        create_sample_data()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    
    if dataset_downloaded:
        print("✓ Project is ready to use")
        print("\nNext steps:")
        print("  1. Run full analysis: python src/main.py")
        print("  2. Run with sample data: python src/main.py --sample-size 1000")
        print("  3. Optimize clusters: python src/main.py --optimize-clusters")
    else:
        print("⚠️  Please download the dataset manually:")
        print("  1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("  2. Download creditcard.csv")
        print("  3. Place it in data/raw/creditcard.csv")
        print("  4. Run: python src/main.py")


if __name__ == "__main__":
    main() 