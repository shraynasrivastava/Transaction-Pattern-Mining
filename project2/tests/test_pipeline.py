import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_processor import DataProcessor
from src.models.fraud_detector import FraudDetector
from src.visualization.visualizer import FraudVisualizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def test_data_processing():
    """Test data loading and processing."""
    processor = DataProcessor()
    
    # Test data loading
    success = processor.load_data()
    assert success, "Failed to load data"
    logger.info("Data loading test passed")
    
    # Log data distribution
    fraud_count = len(processor.data[processor.data['Class'] == 1])
    normal_count = len(processor.data[processor.data['Class'] == 0])
    logger.info(f"Data distribution - Fraud: {fraud_count}, Normal: {normal_count}")
    
    # Test data preprocessing
    features, scaled_features = processor.preprocess_data()
    assert features is not None and scaled_features is not None, "Failed to preprocess data"
    logger.info("Data preprocessing test passed")
    
    # Test SMOTE application with balanced subset using raw features
    # Get all fraud cases and equal number of normal cases
    fraud_data = processor.data[processor.data['Class'] == 1]
    normal_data = processor.data[processor.data['Class'] == 0].sample(n=len(fraud_data), random_state=42)
    combined_data = pd.concat([fraud_data, normal_data])
    
    # Prepare features for SMOTE
    feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount']
    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(combined_data[feature_cols]),
        columns=feature_cols
    )
    y = combined_data['Class']
    
    X_resampled, y_resampled = processor.apply_smote(X, y)
    assert X_resampled is not None and y_resampled is not None, "Failed to apply SMOTE"
    logger.info("SMOTE application test passed")
    
    return processor, scaled_features

def test_model_training(scaled_features):
    """Test model training and risk assignment."""
    detector = FraudDetector()
    
    # Test anomaly detection
    anomaly_scores = detector.detect_anomalies(scaled_features)
    assert anomaly_scores is not None, "Failed to detect anomalies"
    logger.info("Anomaly detection test passed")
    
    # Test clustering with optimization
    cluster_labels = detector.cluster_users(scaled_features, optimize_clusters=True)
    assert cluster_labels is not None, "Failed to perform clustering"
    logger.info("Clustering test passed")
    
    # Test risk label assignment
    risk_labels = detector.assign_risk_labels(anomaly_scores, cluster_labels)
    assert risk_labels is not None, "Failed to assign risk labels"
    logger.info("Risk label assignment test passed")
    
    # Test model saving
    success = detector.save_models()
    assert success, "Failed to save models"
    logger.info("Model saving test passed")
    
    return detector, risk_labels

def test_visualization(processor, detector, risk_labels):
    """Test visualization generation."""
    visualizer = FraudVisualizer()
    
    # Test basic visualizations
    visualizer.plot_fraud_distribution(processor.data)
    visualizer.plot_amount_distribution(processor.data)
    visualizer.plot_time_patterns(processor.data)
    logger.info("Basic visualizations test passed")
    
    # Test feature importance plot
    visualizer.plot_feature_importance(processor.user_features)
    logger.info("Feature importance plot test passed")
    
    # Test cluster visualization
    visualizer.plot_clusters(
        processor.user_features_scaled,
        detector.cluster_labels,
        risk_labels
    )
    logger.info("Cluster visualization test passed")
    
    # Test risk distribution plot
    visualizer.plot_risk_distribution(risk_labels)
    logger.info("Risk distribution plot test passed")
    
    # Test summary report generation
    data_stats = processor.get_data_stats()
    model_summary = detector.get_model_summary()
    visualizer.create_summary_report(
        data_stats,
        model_summary,
        pd.Series(risk_labels)
    )
    logger.info("Summary report generation test passed")

def main():
    """Run all tests."""
    logger.info("Starting pipeline tests...")
    
    try:
        # Test data processing
        processor, scaled_features = test_data_processing()
        
        # Test model training
        detector, risk_labels = test_model_training(scaled_features)
        
        # Test visualization
        test_visualization(processor, detector, risk_labels)
        
        logger.info("All tests passed successfully!")
        
    except AssertionError as e:
        logger.error(f"Test failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main() 