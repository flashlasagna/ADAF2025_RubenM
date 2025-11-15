"""
Visualization functions for model evaluation.
Creates plots for report and presentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import FIGURES_DIR, COLOR_PALETTE

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_model_comparison_regression(results_df: pd.DataFrame,
                                     metric: str = 'rmse',
                                     save_path: Path = None):
    """
    Plot regression model comparison bar chart.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from evaluate_regression_models
    metric : str
        Metric to plot ('rmse', 'mae', 'r2')
    save_path : Path, optional
        Path to save figure
    """
    # Filter to test set
    test_df = results_df[results_df['dataset'] == 'test'].copy()
    test_df = test_df.sort_values(metric, ascending=(metric != 'r2'))

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLOR_PALETTE.get(model, '#888888') for model in test_df['model']]

    ax.barh(test_df['model'], test_df[metric], color=colors)

    # Labels
    metric_labels = {
        'rmse': 'RMSE (°C)',
        'mae': 'MAE (°C)',
        'r2': 'R² Score',
        'mape': 'MAPE (%)'
    }

    ax.set_xlabel(metric_labels.get(metric, metric.upper()))
    ax.set_ylabel('Model')
    ax.set_title(f'Temperature Prediction: {metric_labels.get(metric, metric.upper())} Comparison (Test Set)')

    # Add value labels
    for i, (idx, row) in enumerate(test_df.iterrows()):
        value = row[metric]
        ax.text(value, i, f'  {value:.3f}', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"✓ Saved plot: {save_path}")

    plt.close()


def plot_model_comparison_classification(results_df: pd.DataFrame,
                                         metric: str = 'f1',
                                         save_path: Path = None):
    """
    Plot classification model comparison bar chart.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from evaluate_classification_models
    metric : str
        Metric to plot ('f1', 'precision', 'recall', 'roc_auc')
    save_path : Path, optional
        Path to save figure
    """
    # Filter to test set
    test_df = results_df[results_df['dataset'] == 'test'].copy()
    test_df = test_df.sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLOR_PALETTE.get(model, '#888888') for model in test_df['model']]

    ax.barh(test_df['model'], test_df[metric], color=colors)

    # Labels
    metric_labels = {
        'f1': 'F1-Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'roc_auc': 'ROC-AUC',
        'accuracy': 'Accuracy'
    }

    ax.set_xlabel(metric_labels.get(metric, metric.upper()))
    ax.set_ylabel('Model')
    ax.set_title(f'Rain Prediction: {metric_labels.get(metric, metric.upper())} Comparison (Test Set)')
    ax.set_xlim(0, 1)

    # Add value labels
    for i, (idx, row) in enumerate(test_df.iterrows()):
        value = row[metric]
        ax.text(value, i, f'  {value:.4f}', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"✓ Saved plot: {save_path}")

    plt.close()


def plot_all_metrics_comparison(results_df: pd.DataFrame,
                                task: str = 'regression',
                                save_path: Path = None):
    """
    Plot all metrics for all models in one figure.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe
    task : str
        'regression' or 'classification'
    save_path : Path, optional
        Path to save figure
    """
    # Filter to test set
    test_df = results_df[results_df['dataset'] == 'test'].copy()

    if task == 'regression':
        metrics = ['rmse', 'mae', 'r2']
        titles = ['RMSE (°C)', 'MAE (°C)', 'R² Score']
    else:
        metrics = ['f1', 'precision', 'recall', 'roc_auc']
        titles = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC']

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        df_metric = test_df.sort_values(metric, ascending=(metric == 'rmse' or metric == 'mae'))

        colors = [COLOR_PALETTE.get(model, '#888888') for model in df_metric['model']]

        ax.barh(df_metric['model'], df_metric[metric], color=colors)
        ax.set_xlabel(title)
        ax.set_ylabel('Model' if ax == axes[0] else '')
        ax.set_title(title)

        # Add value labels
        for i, (idx, row) in enumerate(df_metric.iterrows()):
            value = row[metric]
            if metric in ['rmse', 'mae']:
                ax.text(value, i, f'  {value:.3f}', va='center', fontsize=8)
            else:
                ax.text(value, i, f'  {value:.4f}', va='center', fontsize=8)

    task_name = 'Temperature Prediction' if task == 'regression' else 'Rain Prediction'
    fig.suptitle(f'{task_name}: Model Comparison (Test Set)', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"✓ Saved plot: {save_path}")

    plt.close()


def create_all_plots(reg_results: pd.DataFrame,
                     clf_results: pd.DataFrame):
    """
    Create all evaluation plots.

    Parameters
    ----------
    reg_results : pd.DataFrame
        Regression results
    clf_results : pd.DataFrame
        Classification results
    """
    logger.info("\n" + "=" * 60)
    logger.info("CREATING EVALUATION PLOTS")
    logger.info("=" * 60)

    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Regression plots
    logger.info("\nCreating regression plots...")
    plot_model_comparison_regression(reg_results, 'rmse',
                                     FIGURES_DIR / 'regression_rmse_comparison.png')
    plot_model_comparison_regression(reg_results, 'mae',
                                     FIGURES_DIR / 'regression_mae_comparison.png')
    plot_model_comparison_regression(reg_results, 'r2',
                                     FIGURES_DIR / 'regression_r2_comparison.png')
    plot_all_metrics_comparison(reg_results, 'regression',
                                FIGURES_DIR / 'regression_all_metrics.png')

    # Classification plots
    logger.info("\nCreating classification plots...")
    plot_model_comparison_classification(clf_results, 'f1',
                                         FIGURES_DIR / 'classification_f1_comparison.png')
    plot_model_comparison_classification(clf_results, 'roc_auc',
                                         FIGURES_DIR / 'classification_rocauc_comparison.png')
    plot_all_metrics_comparison(clf_results, 'classification',
                                FIGURES_DIR / 'classification_all_metrics.png')

    logger.info(f"\n✓ All plots saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load results (if available)
    from src.utils.config import TABLES_DIR

    reg_results_path = TABLES_DIR / 'regression_results.csv'
    clf_results_path = TABLES_DIR / 'classification_results.csv'

    if reg_results_path.exists() and clf_results_path.exists():
        logger.info("Loading results and creating plots...")
        reg_results = pd.read_csv(reg_results_path)
        clf_results = pd.read_csv(clf_results_path)

        create_all_plots(reg_results, clf_results)
        print("\n✓ Visualization complete!")
    else:
        logger.warning("Results files not found. Run evaluate_models.py first.")