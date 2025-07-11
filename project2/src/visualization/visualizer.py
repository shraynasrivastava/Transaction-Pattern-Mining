import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
from sklearn.decomposition import PCA

from src.config.config import (
    PLOT_STYLE,
    FIGURE_SIZE,
    DPI,
    PROCESSED_DATA_PATH
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FraudVisualizer:
    def __init__(self):
        """Initialize the FraudVisualizer with enhanced styling."""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.output_dir = PROCESSED_DATA_PATH / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set custom color palette
        self.colors = {
            'normal': '#2ecc71',
            'fraud': '#e74c3c',
            'low_risk': '#2ecc71',
            'medium_risk': '#f1c40f',
            'high_risk': '#e74c3c'
        }
        
        # Set default style parameters
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
    def save_plot(self, name: str):
        """Save the current plot with timestamp and enhanced quality."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{name}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Plot saved to {filename}")
        
    def plot_fraud_distribution(self, data: pd.DataFrame):
        """Plot enhanced distribution of fraudulent vs normal transactions."""
        plt.figure(figsize=(12, 6))
        
        # Calculate percentages
        total = len(data)
        normal = len(data[data['Class'] == 0])
        fraud = len(data[data['Class'] == 1])
        
        # Create bar plot
        ax = sns.barplot(x=['Normal', 'Fraud'], 
                        y=[normal, fraud],
                        palette=[self.colors['normal'], self.colors['fraud']])
        
        # Add percentage labels
        for i, v in enumerate([normal, fraud]):
            percentage = v / total * 100
            ax.text(i, v, f'{v:,}\n({percentage:.2f}%)', 
                   ha='center', va='bottom')
        
        plt.title('Distribution of Transaction Classes', pad=20)
        plt.xlabel('Transaction Type')
        plt.ylabel('Number of Transactions')
        
        # Add grid for better readability
        plt.grid(True, axis='y', alpha=0.3)
        
        self.save_plot('fraud_distribution')
        
    def plot_amount_distribution(self, data: pd.DataFrame):
        """Plot enhanced distribution of transaction amounts."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot normal transactions with better binning
        sns.histplot(data=data[data['Class'] == 0]['Amount'], 
                    bins=50, 
                    ax=ax1,
                    color=self.colors['normal'],
                    stat='density',
                    element='step',
                    fill=True,
                    alpha=0.5)
        ax1.set_title('Amount Distribution - Normal Transactions')
        ax1.set_xlabel('Amount ($)')
        ax1.set_ylabel('Density')
        
        # Add statistics for normal transactions
        normal_stats = data[data['Class'] == 0]['Amount'].describe()
        stats_text = f'Mean: ${normal_stats["mean"]:.2f}\n'
        stats_text += f'Median: ${normal_stats["50%"]:.2f}\n'
        stats_text += f'Max: ${normal_stats["max"]:.2f}'
        ax1.text(0.95, 0.95, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot fraudulent transactions
        sns.histplot(data=data[data['Class'] == 1]['Amount'], 
                    bins=50, 
                    ax=ax2,
                    color=self.colors['fraud'],
                    stat='density',
                    element='step',
                    fill=True,
                    alpha=0.5)
        ax2.set_title('Amount Distribution - Fraudulent Transactions')
        ax2.set_xlabel('Amount ($)')
        ax2.set_ylabel('Density')
        
        # Add statistics for fraudulent transactions
        fraud_stats = data[data['Class'] == 1]['Amount'].describe()
        stats_text = f'Mean: ${fraud_stats["mean"]:.2f}\n'
        stats_text += f'Median: ${fraud_stats["50%"]:.2f}\n'
        stats_text += f'Max: ${fraud_stats["max"]:.2f}'
        ax2.text(0.95, 0.95, stats_text,
                transform=ax2.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        self.save_plot('amount_distribution')
        
    def plot_time_patterns(self, data: pd.DataFrame):
        """Plot enhanced transaction patterns over time."""
        plt.figure(figsize=(15, 7))
        
        # Convert time to hour of day
        hour_data = data.copy()
        hour_data['Hour'] = hour_data['Time'].apply(lambda x: (x / 3600) % 24)
        
        # Create separate plots for better visibility
        normal_hours = hour_data[hour_data['Class'] == 0]['Hour']
        fraud_hours = hour_data[hour_data['Class'] == 1]['Hour']
        
        # Plot with transparency and better binning
        plt.hist([normal_hours, fraud_hours], 
                bins=24, 
                label=['Normal', 'Fraud'],
                color=[self.colors['normal'], self.colors['fraud']],
                alpha=0.6,
                density=True,
                stacked=False)
        
        plt.title('Transaction Frequency by Hour of Day', pad=20)
        plt.xlabel('Hour of Day')
        plt.ylabel('Density')
        plt.legend()
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Add hour markers
        plt.xticks(range(0, 24, 2))
        
        self.save_plot('time_patterns')
        
    def plot_feature_importance(self, data: pd.DataFrame):
        """Plot enhanced correlation matrix of features."""
        plt.figure(figsize=(15, 12))
        
        # Calculate correlation matrix
        correlation_matrix = data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Plot correlation heatmap with improved styling
        sns.heatmap(correlation_matrix,
                   mask=mask,
                   annot=True,
                   cmap='RdYlBu_r',
                   center=0,
                   fmt='.2f',
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": .5})
        
        plt.title('Feature Correlation Matrix', pad=20)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        self.save_plot('feature_correlations')
        
    def plot_clusters(self, 
                     data: pd.DataFrame, 
                     labels: np.ndarray, 
                     risk_labels: Optional[pd.Series] = None):
        """Plot enhanced cluster visualization using PCA."""
        # Reduce dimensions for visualization
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        
        plt.figure(figsize=(12, 8))
        
        if risk_labels is not None:
            # Create color map for risk levels
            risk_colors = [self.colors[label] for label in risk_labels]
            
            # Plot with risk labels
            scatter = plt.scatter(data_2d[:, 0],
                                data_2d[:, 1],
                                c=risk_colors,
                                alpha=0.6)
            
            # Create custom legend
            legend_elements = [plt.Line2D([0], [0],
                                        marker='o',
                                        color='w',
                                        markerfacecolor=color,
                                        label=label.replace('_', ' ').title(),
                                        markersize=10)
                             for label, color in self.colors.items()
                             if label in ['low_risk', 'medium_risk', 'high_risk']]
            plt.legend(handles=legend_elements)
        else:
            # Plot with cluster labels using a different colormap
            scatter = plt.scatter(data_2d[:, 0],
                                data_2d[:, 1],
                                c=labels,
                                cmap='viridis',
                                alpha=0.6)
            plt.colorbar(scatter, label='Cluster')
        
        plt.title('Transaction Clusters Visualization', pad=20)
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        self.save_plot('cluster_visualization')
        
    def plot_risk_distribution(self, risk_labels: pd.Series):
        """Plot enhanced distribution of risk labels."""
        plt.figure(figsize=(12, 6))
        
        risk_counts = risk_labels.value_counts()
        total = risk_counts.sum()
        
        # Create bar plot with enhanced styling
        bars = plt.bar(range(len(risk_counts)),
                      risk_counts.values,
                      color=[self.colors[label] for label in risk_counts.index])
        
        # Add percentage labels on top of bars
        for i, (label, count) in enumerate(risk_counts.items()):
            percentage = count / total * 100
            plt.text(i, count, f'{count:,}\n({percentage:.1f}%)',
                    ha='center', va='bottom')
        
        plt.title('Distribution of Risk Labels', pad=20)
        plt.xlabel('Risk Level')
        plt.ylabel('Number of Transactions')
        
        # Customize x-axis labels
        plt.xticks(range(len(risk_counts)),
                  [label.replace('_', ' ').title() for label in risk_counts.index])
        
        # Add grid for better readability
        plt.grid(True, axis='y', alpha=0.3)
        
        self.save_plot('risk_distribution')
        
    def create_summary_report(self, 
                            data_stats: dict, 
                            model_summary: dict,
                            risk_distribution: pd.Series):
        """Create an enhanced summary report with key findings."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"summary_report_{timestamp}.txt"
        
        def format_dict(d: dict, indent: int = 0) -> str:
            """Helper function to format nested dictionaries."""
            lines = []
            prefix = ' ' * indent
            for k, v in d.items():
                if isinstance(v, dict):
                    lines.append(f"{prefix}{k}:")
                    lines.append(format_dict(v, indent + 2))
                elif isinstance(v, float):
                    lines.append(f"{prefix}{k}: {v:.4f}")
                else:
                    lines.append(f"{prefix}{k}: {v}")
            return '\n'.join(lines)
        
        with open(report_path, 'w') as f:
            f.write("Fraud Detection Analysis Summary Report\n")
            f.write("=====================================\n\n")
            
            # Data Statistics
            f.write("1. Data Overview\n")
            f.write("--------------\n")
            f.write(format_dict(data_stats, 2))
            f.write("\n\n")
            
            # Model Information
            f.write("2. Model Information\n")
            f.write("------------------\n")
            f.write(format_dict(model_summary, 2))
            f.write("\n\n")
            
            # Risk Distribution
            f.write("3. Risk Distribution\n")
            f.write("------------------\n")
            risk_counts = risk_distribution.value_counts()
            total = risk_counts.sum()
            for label, count in risk_counts.items():
                percentage = count / total * 100
                f.write(f"  {label}: {count:,} ({percentage:.2f}%)\n")
            
            # Save as JSON for programmatic access
            json_path = self.output_dir / f"analysis_summary_{timestamp}.json"
            pd.Series({
                'data_stats': data_stats,
                'model_summary': model_summary,
                'risk_distribution': risk_counts.to_dict()
            }).to_json(json_path)
            
        logger.info(f"Summary report saved to {report_path}")
        logger.info(f"Analysis summary saved to {json_path}") 