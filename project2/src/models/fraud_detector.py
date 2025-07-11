import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from datetime import datetime

from src.config.config import (
    MODEL_ARTIFACTS_PATH,
    ISOLATION_FOREST_PARAMS,
    KMEANS_PARAMS,
    RISK_LABELS,
    ANOMALY_THRESHOLD
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FraudDetector:
    def __init__(self):
        """Initialize the FraudDetector with anomaly detection and clustering models."""
        self.isolation_forest = IsolationForest(**ISOLATION_FOREST_PARAMS)
        self.kmeans = KMeans(**KMEANS_PARAMS)
        self.anomaly_scores = None
        self.cluster_labels = None
        self.risk_labels = None
        self.data = None
        
    def find_optimal_clusters(self, data: pd.DataFrame, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            data: Feature matrix
            max_clusters: Maximum number of clusters to try
            
        Returns:
            int: Optimal number of clusters
        """
        logger.info("Finding optimal number of clusters...")
        
        silhouette_scores = []
        inertias = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            silhouette_scores.append(silhouette_score(data, labels))
            inertias.append(kmeans.inertia_)
            
        # Find the elbow point
        diffs = np.diff(inertias)
        elbow_point = np.argmin(np.abs(diffs - np.mean(diffs))) + 2
        
        # Find the best silhouette score
        best_silhouette = np.argmax(silhouette_scores) + 2
        
        # Take the average of both methods
        optimal_clusters = int((elbow_point + best_silhouette) / 2)
        logger.info(f"Optimal number of clusters: {optimal_clusters}")
        
        return optimal_clusters
        
    def detect_anomalies(self, data: pd.DataFrame) -> np.ndarray:
        """Detect anomalies using Isolation Forest.
        
        Args:
            data: Feature matrix
            
        Returns:
            np.ndarray: Anomaly scores
        """
        logger.info("Detecting anomalies using Isolation Forest...")
        
        try:
            self.data = data  # Store data for later use
            self.anomaly_scores = self.isolation_forest.fit_predict(data)
            logger.info("Anomaly detection completed successfully")
            return self.anomaly_scores
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return None
            
    def cluster_users(self, data: pd.DataFrame, optimize_clusters: bool = True) -> np.ndarray:
        """Cluster users based on their behavior.
        
        Args:
            data: Feature matrix
            optimize_clusters: Whether to find optimal number of clusters
            
        Returns:
            np.ndarray: Cluster labels
        """
        try:
            if optimize_clusters:
                optimal_k = self.find_optimal_clusters(data)
                self.kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                
            logger.info(f"Clustering users with {self.kmeans.n_clusters} clusters...")
            self.cluster_labels = self.kmeans.fit_predict(data)
            logger.info("Clustering completed successfully")
            return self.cluster_labels
            
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            return None
            
    def assign_risk_labels(self, anomaly_scores: np.ndarray, cluster_labels: np.ndarray) -> pd.Series:
        """Assign risk labels based on anomaly scores and cluster patterns.
        
        Args:
            anomaly_scores: Array of anomaly scores
            cluster_labels: Array of cluster labels
            
        Returns:
            pd.Series: Risk labels for each user
        """
        try:
            if self.data is None:
                raise ValueError("No data available. Run detect_anomalies first.")
                
            # Calculate cluster risk based on average anomaly score in each cluster
            cluster_risks = pd.DataFrame({
                'cluster': cluster_labels,
                'anomaly_score': anomaly_scores
            }).groupby('cluster')['anomaly_score'].mean()
            
            # Normalize cluster risks to [0,1]
            cluster_risk = (cluster_risks - cluster_risks.min()) / (cluster_risks.max() - cluster_risks.min())
            
            # Create a DataFrame with all scores
            risk_df = pd.DataFrame({
                'anomaly_score': anomaly_scores,
                'cluster': cluster_labels,
                'cluster_risk': cluster_risk[cluster_labels].values
            })
            
            # Initialize all as medium risk
            risk_df['risk_label'] = RISK_LABELS[1]  # medium_risk
            
            # Update high risk cases
            risk_df.loc[
                (risk_df['cluster_risk'] > ANOMALY_THRESHOLD) | 
                (risk_df['anomaly_score'] == -1),
                'risk_label'
            ] = RISK_LABELS[2]  # high_risk
            
            # Update low risk cases
            risk_df.loc[
                (risk_df['cluster_risk'] < 0.5) & 
                (risk_df['anomaly_score'] == 1),
                'risk_label'
            ] = RISK_LABELS[0]  # low_risk
            
            self.risk_labels = risk_df['risk_label']
            return self.risk_labels
            
        except Exception as e:
            logger.error(f"Error in risk label assignment: {str(e)}")
            return None
            
    def save_models(self) -> bool:
        """Save trained models to disk.
        
        Returns:
            bool: True if models were saved successfully, False otherwise
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save Isolation Forest
            isolation_forest_path = MODEL_ARTIFACTS_PATH / f"isolation_forest_{timestamp}.joblib"
            joblib.dump(self.isolation_forest, isolation_forest_path)
            
            # Save KMeans
            kmeans_path = MODEL_ARTIFACTS_PATH / f"kmeans_{timestamp}.joblib"
            joblib.dump(self.kmeans, kmeans_path)
            
            logger.info(f"Models saved successfully to {MODEL_ARTIFACTS_PATH}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
            
    def load_models(self, isolation_forest_path: Path, kmeans_path: Path) -> bool:
        """Load trained models from disk.
        
        Args:
            isolation_forest_path: Path to saved Isolation Forest model
            kmeans_path: Path to saved KMeans model
            
        Returns:
            bool: True if models were loaded successfully, False otherwise
        """
        try:
            self.isolation_forest = joblib.load(isolation_forest_path)
            self.kmeans = joblib.load(kmeans_path)
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
            
    def get_model_summary(self) -> Dict:
        """Get summary of model parameters and performance.
        
        Returns:
            dict: Dictionary containing model information
        """
        if self.anomaly_scores is None or self.cluster_labels is None:
            return {}
            
        summary = {
            'isolation_forest_params': self.isolation_forest.get_params(),
            'kmeans_params': self.kmeans.get_params(),
            'n_clusters': self.kmeans.n_clusters,
            'anomaly_distribution': {
                'normal': np.sum(self.anomaly_scores == 1),
                'anomaly': np.sum(self.anomaly_scores == -1)
            },
            'cluster_sizes': pd.Series(self.cluster_labels).value_counts().to_dict(),
            'risk_distribution': (
                pd.Series(self.risk_labels).value_counts().to_dict() 
                if self.risk_labels is not None else {}
            )
        }
        
        return summary 