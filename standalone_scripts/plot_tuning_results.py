"""
Fixed visualization script for comprehensive tuning results.
Uses correct file names from comprehensive_tuning.py output.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent
TABLES_DIR = PROJECT_ROOT / 'results' / 'tables'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'

# Create figures directory
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300


def create_improvement_comparison_plot(task='regression'):
    """
    Create before/after tuning comparison plot.

    Parameters
    ----------
    task : str
        'regression' or 'classification'
    """
    logger.info(f"Creating improvement comparison for {task}...")

    # Load comprehensive tuning results
    tuning_file = TABLES_DIR / f'hyperparameter_tuning_{task}_comprehensive.csv'
    if not tuning_file.exists():
        logger.error(f"File not found: {tuning_file}")
        return

    tuning_df = pd.read_csv(tuning_file)

    # Load baseline results
    baseline_file = TABLES_DIR / f'{task}_results.csv'
    if not baseline_file.exists():
        logger.error(f"File not found: {baseline_file}")
        return

    baseline_df = pd.read_csv(baseline_file)

    # Get test set results only
    baseline_test = baseline_df[baseline_df['dataset'] == 'test'].copy()

    # Prepare data
    models = tuning_df['Model'].values

    if task == 'regression':
        baseline_scores = baseline_test.set_index('model')['rmse'].reindex(models).values
        tuned_scores = tuning_df['Best Score'].values
        ylabel = 'RMSE (°C) - Lower is Better'
        title = 'Temperature Prediction: Hyperparameter Tuning Impact'
    else:
        baseline_scores = baseline_test.set_index('model')['f1'].reindex(models).values
        tuned_scores = tuning_df['Best Score'].values
        ylabel = 'F1-Score - Higher is Better'
        title = 'Rain Prediction: Hyperparameter Tuning Impact'

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    # Plot bars
    baseline_bars = ax.bar(x - width/2, baseline_scores, width,
                          label='Default Parameters', alpha=0.8, color='#F38181')
    tuned_bars = ax.bar(x + width/2, tuned_scores, width,
                       label='Tuned Parameters', alpha=0.8, color='#95E1D3')

    # Customize
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [baseline_bars, tuned_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    # Add improvement annotations
    for i, (baseline, tuned) in enumerate(zip(baseline_scores, tuned_scores)):
        if pd.notna(baseline) and pd.notna(tuned):
            if task == 'regression':
                improvement = ((baseline - tuned) / baseline) * 100
                y_pos = max(baseline, tuned) * 1.02
            else:
                improvement = ((tuned - baseline) / baseline) * 100
                y_pos = max(baseline, tuned) * 1.02

            color = 'green' if improvement > 0 else 'red'
            ax.text(i, y_pos, f'{improvement:+.1f}%',
                   ha='center', fontsize=9, fontweight='bold', color=color)

    plt.tight_layout()

    # Save
    save_path = FIGURES_DIR / f'{task}_tuning_improvement.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"--OK-- Saved: {save_path}")

    plt.close()


def create_tuned_scores_plot(task='regression'):
    """
    Create bar plot of final tuned scores.

    Parameters
    ----------
    task : str
        'regression' or 'classification'
    """
    logger.info(f"Creating tuned scores plot for {task}...")

    # Load comprehensive tuning results
    tuning_file = TABLES_DIR / f'hyperparameter_tuning_{task}_comprehensive.csv'
    if not tuning_file.exists():
        logger.error(f"File not found: {tuning_file}")
        return

    tuning_df = pd.read_csv(tuning_file)

    # Sort by score
    if task == 'regression':
        tuning_df = tuning_df.sort_values('Best Score', ascending=True)
    else:
        tuning_df = tuning_df.sort_values('Best Score', ascending=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#95E1D3', '#F38181', '#38A3A5', '#22A39F']

    bars = ax.barh(tuning_df['Model'], tuning_df['Best Score'], color=colors)

    if task == 'regression':
        xlabel = 'RMSE (°C) - Lower is Better'
        title = 'Final Temperature Prediction Performance (Tuned Models)'
    else:
        xlabel = 'F1-Score - Higher is Better'
        title = 'Final Rain Prediction Performance (Tuned Models)'

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{width:.4f}',
               ha='left', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save
    save_path = FIGURES_DIR / f'{task}_best_scores_tuned.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"--OK-- Saved: {save_path}")

    plt.close()


def create_summary_table_plot(task='regression'):
    """
    Create a table showing tuning summary.

    Parameters
    ----------
    task : str
        'regression' or 'classification'
    """
    logger.info(f"Creating summary table for {task}...")

    # Load data
    tuning_file = TABLES_DIR / f'hyperparameter_tuning_{task}_comprehensive.csv'
    baseline_file = TABLES_DIR / f'{task}_results.csv'

    if not tuning_file.exists() or not baseline_file.exists():
        logger.warning("Missing files for summary table")
        return

    tuning_df = pd.read_csv(tuning_file)
    baseline_df = pd.read_csv(baseline_file)
    baseline_test = baseline_df[baseline_df['dataset'] == 'test'].copy()

    # Prepare data
    models = tuning_df['Model'].values

    if task == 'regression':
        baseline_scores = baseline_test.set_index('model')['rmse'].reindex(models).values
        tuned_scores = tuning_df['Best Score'].values
        metric_name = 'RMSE (°C)'
    else:
        baseline_scores = baseline_test.set_index('model')['f1'].reindex(models).values
        tuned_scores = tuning_df['Best Score'].values
        metric_name = 'F1-Score'

    # Calculate improvements
    if task == 'regression':
        improvements = ((baseline_scores - tuned_scores) / baseline_scores) * 100
    else:
        improvements = ((tuned_scores - baseline_scores) / baseline_scores) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    for i, model in enumerate(models):
        if pd.notna(baseline_scores[i]):
            table_data.append([
                model.upper(),
                f'{baseline_scores[i]:.4f}',
                f'{tuned_scores[i]:.4f}',
                f'{improvements[i]:+.1f}%'
            ])

    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Model', f'Baseline\n{metric_name}',
                              f'Tuned\n{metric_name}', 'Improvement'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#38A3A5')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style cells
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if j == 3:  # Improvement column
                cell_text = table_data[i-1][j]
                if '+' in cell_text:
                    table[(i, j)].set_facecolor('#E8F8F5')
                else:
                    table[(i, j)].set_facecolor('#FADBD8')

    task_name = 'Temperature Prediction' if task == 'regression' else 'Rain Prediction'
    plt.title(f'{task_name}: Hyperparameter Tuning Results',
             fontsize=14, fontweight='bold', pad=20)

    # Save
    save_path = FIGURES_DIR / f'{task}_tuning_summary_table.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"--OK-- Saved: {save_path}")

    plt.close()


def create_all_plots():
    """Create all visualization plots."""
    logger.info("\n" + "="*80)
    logger.info("CREATING COMPREHENSIVE TUNING VISUALIZATIONS")
    logger.info("="*80)

    # Regression plots
    logger.info("\nRegression plots...")
    create_improvement_comparison_plot(task='regression')
    create_tuned_scores_plot(task='regression')
    create_summary_table_plot(task='regression')

    # Classification plots
    logger.info("\nClassification plots...")
    create_improvement_comparison_plot(task='classification')
    create_tuned_scores_plot(task='classification')
    create_summary_table_plot(task='classification')

    logger.info("\n" + "="*80)
    logger.info("--OK-- ALL PLOTS CREATED!")
    logger.info("="*80)
    logger.info(f"\nPlots saved to: {FIGURES_DIR}")

    # List created plots
    plots = list(FIGURES_DIR.glob('*.png'))
    logger.info(f"\nCreated {len(plots)} plots:")
    for plot in sorted(plots):
        logger.info(f"  - {plot.name}")


if __name__ == "__main__":
    create_all_plots()