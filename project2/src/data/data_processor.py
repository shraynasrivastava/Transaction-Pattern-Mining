import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.config.config import (
    RAW_DATA_PATH, 
    PROCESSED_DATA_PATH, 
    TIME_WINDOW_QUANTILES,
    FEATURE_COLUMNS
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor."""
        self.scaler = StandardScaler()
        self.data = None
        self.user_features = None
        self.user_features_scaled = None
        
    def load_data(self) -> bool:
        """Load the credit card transaction data.
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading data from {RAW_DATA_PATH}")
            self.data = pd.read_csv(RAW_DATA_PATH)
            
            if not all(col in self.data.columns for col in FEATURE_COLUMNS):
                logger.error("Missing required columns in the dataset")
                return False
                
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Data file not found at {RAW_DATA_PATH}")
            return False
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
            
    def preprocess_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Preprocess the data and engineer features.
        
        Returns:
            Tuple containing:
                - Processed user features
                - Scaled user features (if scaling was performed)
        """
        if self.data is None:
            logger.error("No data loaded. Please load data first.")
            return None, None
            
        try:
            logger.info("Starting data preprocessing...")
            
            # Convert time to hour of day
            self.data['Hour'] = self.data['Time'].apply(lambda x: (x / 3600) % 24)
            
            # Create time windows for user segmentation
            time_windows = pd.qcut(self.data['Time'], 
                                 q=TIME_WINDOW_QUANTILES, 
                                 labels=False)
            
            # Engineer user-level features
            self.user_features = pd.DataFrame({
                'avg_amount': self.data.groupby(time_windows)['Amount'].mean(),
                'tx_frequency': self.data.groupby(time_windows).size(),
                'amount_std': self.data.groupby(time_windows)['Amount'].std(),
                'hour_std': self.data.groupby(time_windows)['Hour'].std(),
                'fraud_ratio': self.data.groupby(time_windows)['Class'].mean(),
                'v1_mean': self.data.groupby(time_windows)['V1'].mean(),
                'v2_mean': self.data.groupby(time_windows)['V2'].mean(),
                'v3_mean': self.data.groupby(time_windows)['V3'].mean(),
                'v4_mean': self.data.groupby(time_windows)['V4'].mean()
            }).fillna(0)
            
            # Scale features
            self.user_features_scaled = pd.DataFrame(
                self.scaler.fit_transform(self.user_features),
                columns=self.user_features.columns,
                index=self.user_features.index
            )
            
            # Save processed features
            processed_file = PROCESSED_DATA_PATH / "processed_features.csv"
            self.user_features.to_csv(processed_file)
            logger.info(f"Processed features saved to {processed_file}")
            
            return self.user_features, self.user_features_scaled
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            return None, None
            
    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to handle class imbalance.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuple containing:
                - Resampled feature matrix
                - Resampled target variable
        """
        try:
            logger.info("Applying SMOTE for handling class imbalance...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info("SMOTE applied successfully")
            return X_resampled, y_resampled
        except Exception as e:
            logger.error(f"Error applying SMOTE: {str(e)}")
            return None, None
            
    def get_data_stats(self) -> dict:
        """Get basic statistics about the data.
        
        Returns:
            dict: Dictionary containing data statistics
        """
        if self.data is None:
            return {}
            
        stats = {
            'total_transactions': len(self.data),
            'fraud_transactions': len(self.data[self.data['Class'] == 1]),
            'normal_transactions': len(self.data[self.data['Class'] == 0]),
            'fraud_ratio': len(self.data[self.data['Class'] == 1]) / len(self.data),
            'amount_stats': self.data['Amount'].describe().to_dict(),
            'time_range': {
                'start': self.data['Time'].min(),
                'end': self.data['Time'].max()
            }
        }
        
        return stats 