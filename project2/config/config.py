from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
MODEL_ARTIFACTS_PATH = ROOT_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

# Model parameters
ISOLATION_FOREST_PARAMS = {
    'contamination': 0.02,
    'random_state': 42,
    'n_estimators': 100,
    'max_samples': 'auto'
}

KMEANS_PARAMS = {
    'n_clusters': 3,
    'random_state': 42,
    'n_init': 10
}

# Feature engineering parameters
TIME_WINDOW_QUANTILES = 1000
FEATURE_COLUMNS = [
    'Time', 'Amount', 'Class',
    *[f'V{i}' for i in range(1, 29)]
]

# Visualization settings
PLOT_STYLE = 'seaborn'
FIGURE_SIZE = (12, 8)
DPI = 300

# Logging configuration
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "fraud_detection.log"

# Model evaluation
RISK_LABELS = ['low_risk', 'medium_risk', 'high_risk']
ANOMALY_THRESHOLD = 0.98  # 98th percentile for anomaly scores 