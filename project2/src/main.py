#!/usr/bin/env python3
"""
Credit Card Fraud Detection System - Main Entry Point

This is the main script that orchestrates the entire fraud detection pipeline:
1. Data loading and preprocessing
2. Model training (Isolation Forest + KMeans)
3. Risk scoring and analysis
4. Visualization generation
5. Report creation

Author: Your Name
Date: 2024
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_processor import DataProcessor
from models.fraud_detector import FraudDetector
from visualization.visualizer import Visualizer
from utils.logger import setup_logging


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found!")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


def ensure_directories(config):
    """Ensure all required directories exist."""
    directories = [
        config['data']['processed_data_path'],
        config['data']['sample_data_path'],
        config['models']['save_path'],
        config['visualization']['output_path'],
        config['output']['reports_path'],
        os.path.dirname(config['logging']['log_file'])
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection System')
    parser.add_argument('--config', default='config/config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--load-models', action='store_true',
                       help='Load existing models instead of training new ones')
    parser.add_argument('--optimize-clusters', action='store_true',
                       help='Optimize number of clusters using elbow method')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Use only a sample of the data for testing')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.load_models:
        config['models']['load_existing'] = True
    if args.optimize_clusters:
        config['models']['kmeans']['optimize_clusters'] = True
    
    # Ensure directories exist
    ensure_directories(config)
    
    # Setup logging
    setup_logging(config['logging'])
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Credit Card Fraud Detection Pipeline")
        logger.info(f"Configuration loaded from: {args.config}")
        
        # Initialize components
        print("Initializing components...")
        data_processor = DataProcessor(config)
        fraud_detector = FraudDetector(config)
        visualizer = Visualizer(config)
        
        # Step 1: Load and preprocess data
        print("Loading and preprocessing data...")
        logger.info("Loading and preprocessing data")
        
        if not os.path.exists(config['data']['raw_data_path']):
            logger.error(f"Raw data file not found: {config['data']['raw_data_path']}")
            print(f"Error: Please ensure {config['data']['raw_data_path']} exists.")
            print("You can download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            sys.exit(1)
        
        data = data_processor.load_data()
        
        if args.sample_size:
            print(f"Using sample of {args.sample_size} records for testing...")
            data = data.sample(n=min(args.sample_size, len(data)), random_state=42)
        
        processed_data = data_processor.preprocess_data(data)
        
        # Step 2: Train or load models
        print("Training/Loading models...")
        logger.info("Training or loading fraud detection models")
        
        models = fraud_detector.train_models(processed_data)
        
        # Step 3: Perform fraud detection and clustering
        print("Performing fraud detection and risk analysis...")
        logger.info("Performing fraud detection and clustering analysis")
        
        results = fraud_detector.predict_and_analyze(processed_data)
        
        # Step 4: Generate visualizations
        print("Generating visualizations...")
        logger.info("Generating visualization reports")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualizations = visualizer.generate_all_plots(
            data=processed_data,
            results=results,
            timestamp=timestamp
        )
        
        # Step 5: Generate summary report
        print("Generating analysis report...")
        logger.info("Generating summary analysis report")
        
        report_path = fraud_detector.generate_report(
            data=processed_data,
            results=results,
            visualizations=visualizations,
            timestamp=timestamp
        )
        
        # Summary
        print("\n" + "="*60)
        print("FRAUD DETECTION ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total transactions analyzed: {len(processed_data):,}")
        print(f"Fraudulent transactions detected: {results['fraud_count']:,}")
        print(f"Fraud rate: {results['fraud_rate']:.2%}")
        print(f"Number of risk clusters: {results['n_clusters']}")
        print(f"\nOutputs saved to:")
        print(f"  - Visualizations: {config['visualization']['output_path']}")
        print(f"  - Analysis report: {report_path}")
        print(f"  - Models: {config['models']['save_path']}")
        print(f"  - Logs: {config['logging']['log_file']}")
        
        logger.info("Pipeline completed successfully")
        print("\nPipeline completed successfully! ✅")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 