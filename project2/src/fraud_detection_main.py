import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    def __init__(self, data_path):
        """Initialize the Fraud Detection System"""
        self.data_path = data_path
        self.data = None
        self.scaler = MinMaxScaler()
        self.isolation_forest = None
        self.kmeans = None
        self.pca = None
        
    def load_data(self):
        """Load and perform initial data checks"""
        print("Loading data...")
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return True
        except FileNotFoundError:
            print(f"Error: The file {self.data_path} was not found.")
            print("Please download the dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            print("and place it in the same directory as this script.")
            return False
            
    def explore_data(self):
        """Perform initial data exploration"""
        if self.data is None:
            print("Please load the data first using load_data()")
            return
            
        print("\nData Exploration:")
        print("-----------------")
        print(f"Total Transactions: {len(self.data)}")
        print(f"Fraudulent Transactions: {len(self.data[self.data['Class'] == 1])}")
        print(f"Valid Transactions: {len(self.data[self.data['Class'] == 0])}")
        
        # Plot fraud distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data, x='Class')
        plt.title('Distribution of Fraud vs Normal Transactions')
        plt.savefig('fraud_distribution.png')
        plt.close()
        
        # Amount distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.data, x='Amount', hue='Class', bins=50)
        plt.title('Transaction Amount Distribution')
        plt.savefig('amount_distribution.png')
        plt.close()
        
    def engineer_features(self):
        """Perform feature engineering"""
        if self.data is None:
            print("Please load the data first using load_data()")
            return
            
        # Convert time to hour of day
        self.data['Hour'] = self.data['Time'].apply(lambda x: (x / 3600) % 24)
        
        # Aggregate features per user (using Time as proxy for user ID for this example)
        time_windows = pd.qcut(self.data['Time'], q=1000, labels=False)  # Create proxy user IDs
        
        user_features = pd.DataFrame({
            'avg_amount': self.data.groupby(time_windows)['Amount'].mean(),
            'tx_frequency': self.data.groupby(time_windows).size(),
            'amount_std': self.data.groupby(time_windows)['Amount'].std(),
            'hour_std': self.data.groupby(time_windows)['Hour'].std(),
            'fraud_ratio': self.data.groupby(time_windows)['Class'].mean()
        }).fillna(0)
        
        # Normalize features
        self.user_features_scaled = pd.DataFrame(
            self.scaler.fit_transform(user_features),
            columns=user_features.columns,
            index=user_features.index
        )
        
        return self.user_features_scaled
        
    def detect_anomalies(self):
        """Perform anomaly detection using Isolation Forest"""
        if not hasattr(self, 'user_features_scaled'):
            print("Please run engineer_features() first")
            return
            
        print("\nPerforming Anomaly Detection...")
        self.isolation_forest = IsolationForest(contamination=0.02, random_state=42)
        anomaly_scores = self.isolation_forest.fit_predict(self.user_features_scaled)
        
        # Convert predictions to risk labels (-1 for anomaly, 1 for normal)
        self.risk_labels = pd.Series(
            ['high_risk' if score == -1 else 'normal' for score in anomaly_scores],
            index=self.user_features_scaled.index
        )
        
        # Save the model
        joblib.dump(self.isolation_forest, 'isolation_forest_model.joblib')
        
        # Plot anomaly distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.risk_labels)
        plt.title('Distribution of Risk Labels')
        plt.savefig('risk_distribution.png')
        plt.close()
        
        return self.risk_labels
        
    def cluster_users(self, n_clusters=3):
        """Perform clustering on users"""
        if not hasattr(self, 'user_features_scaled'):
            print("Please run engineer_features() first")
            return
            
        print("\nPerforming Clustering...")
        # Reduce dimensions for visualization
        self.pca = PCA(n_components=2)
        user_features_2d = self.pca.fit_transform(self.user_features_scaled)
        
        # Perform KMeans clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(self.user_features_scaled)
        
        # Plot clusters
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(user_features_2d[:, 0], user_features_2d[:, 1], 
                            c=cluster_labels, cmap='viridis')
        plt.title('User Clusters')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        plt.savefig('user_clusters.png')
        plt.close()
        
        # Save the model
        joblib.dump(self.kmeans, 'kmeans_model.joblib')
        
        return cluster_labels
        
    def evaluate_results(self):
        """Evaluate the results and generate risk labels"""
        if not hasattr(self, 'risk_labels'):
            print("Please run detect_anomalies() first")
            return
            
        # Combine anomaly detection and clustering results
        results_df = pd.DataFrame({
            'risk_label': self.risk_labels,
            'cluster': self.kmeans.labels_ if self.kmeans is not None else None,
            'fraud_ratio': self.user_features_scaled['fraud_ratio']
        })
        
        # Export results
        results_df.to_csv('risk_labels.csv')
        
        print("\nEvaluation Results:")
        print("-----------------")
        print(f"High Risk Users: {len(results_df[results_df['risk_label'] == 'high_risk'])}")
        print(f"Normal Users: {len(results_df[results_df['risk_label'] == 'normal'])}")
        
        return results_df

def main():
    # Initialize the system
    fraud_detector = FraudDetectionSystem('creditcard.csv')
    
    # Run the pipeline
    if fraud_detector.load_data():
        fraud_detector.explore_data()
        fraud_detector.engineer_features()
        fraud_detector.detect_anomalies()
        fraud_detector.cluster_users()
        results = fraud_detector.evaluate_results()
        
        print("\nAnalysis complete! Check the generated visualizations and risk_labels.csv for results.")
    
if __name__ == "__main__":
    main() 