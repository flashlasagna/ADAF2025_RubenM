"""
Statistical tests for model comparison.
Tests whether performance differences are statistically significant.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def paired_t_test(errors1: np.ndarray,
                 errors2: np.ndarray,
                 model1_name: str,
                 model2_name: str) -> Dict:
    """
    Perform paired t-test on prediction errors.

    Tests null hypothesis: mean(|errors1|) = mean(|errors2|)

    Parameters
    ----------
    errors1 : np.ndarray
        Absolute errors from model 1
    errors2 : np.ndarray
        Absolute errors from model 2
    model1_name : str
        Name of model 1
    model2_name : str
        Name of model 2

    Returns
    -------
    dict
        Test results (t_statistic, p_value, significant, winner)
    """
    # Use absolute errors
    abs_errors1 = np.abs(errors1)
    abs_errors2 = np.abs(errors2)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(abs_errors1, abs_errors2)

    # Determine significance (α = 0.05)
    significant = p_value < 0.05

    # Determine winner (lower mean error is better)
    mean1 = np.mean(abs_errors1)
    mean2 = np.mean(abs_errors2)
    winner = model1_name if mean1 < mean2 else model2_name

    results = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': significant,
        'winner': winner,
        'mean_error_1': mean1,
        'mean_error_2': mean2
    }

    logger.info(f"\nPaired t-test: {model1_name} vs {model2_name}")
    logger.info(f"  Mean error {model1_name}: {mean1:.4f}")
    logger.info(f"  Mean error {model2_name}: {mean2:.4f}")
    logger.info(f"  t-statistic: {t_stat:.4f}")
    logger.info(f"  p-value: {p_value:.6f}")

    if significant:
        logger.info(f"  ✓ Difference is statistically SIGNIFICANT (p < 0.05)")
        logger.info(f"  Winner: {winner}")
    else:
        logger.info(f"  ✗ Difference is NOT statistically significant (p >= 0.05)")

    return results


def wilcoxon_test(errors1: np.ndarray,
                 errors2: np.ndarray,
                 model1_name: str,
                 model2_name: str) -> Dict:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to t-test).

    Better when errors are not normally distributed.

    Parameters
    ----------
    errors1 : np.ndarray
        Absolute errors from model 1
    errors2 : np.ndarray
        Absolute errors from model 2
    model1_name : str
        Name of model 1
    model2_name : str
        Name of model 2

    Returns
    -------
    dict
        Test results
    """
    abs_errors1 = np.abs(errors1)
    abs_errors2 = np.abs(errors2)

    # Wilcoxon test
    statistic, p_value = stats.wilcoxon(abs_errors1, abs_errors2)

    significant = p_value < 0.05

    mean1 = np.mean(abs_errors1)
    mean2 = np.mean(abs_errors2)
    winner = model1_name if mean1 < mean2 else model2_name

    results = {
        'statistic': statistic,
        'p_value': p_value,
        'significant': significant,
        'winner': winner
    }

    logger.info(f"\nWilcoxon test: {model1_name} vs {model2_name}")
    logger.info(f"  Statistic: {statistic:.2f}")
    logger.info(f"  p-value: {p_value:.6f}")
    logger.info(f"  Significant: {significant}")

    return results


def mcnemar_test(y_true: np.ndarray,
                y_pred1: np.ndarray,
                y_pred2: np.ndarray,
                model1_name: str,
                model2_name: str) -> Dict:
    """
    McNemar's test for comparing two classifiers.

    Tests whether two models make different errors.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred1 : np.ndarray
        Predictions from model 1
    y_pred2 : np.ndarray
        Predictions from model 2
    model1_name : str
        Name of model 1
    model2_name : str
        Name of model 2

    Returns
    -------
    dict
        Test results
    """
    # Create contingency table
    # Count cases where models disagree
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    # n01: model1 wrong, model2 correct
    n01 = np.sum(~correct1 & correct2)
    # n10: model1 correct, model2 wrong
    n10 = np.sum(correct1 & ~correct2)

    # McNemar statistic with continuity correction
    statistic = (np.abs(n10 - n01) - 1)**2 / (n10 + n01) if (n10 + n01) > 0 else 0

    # Chi-square test with 1 df
    p_value = 1 - stats.chi2.cdf(statistic, 1)

    significant = p_value < 0.05

    acc1 = np.mean(correct1)
    acc2 = np.mean(correct2)
    winner = model1_name if acc1 > acc2 else model2_name

    results = {
        'statistic': statistic,
        'p_value': p_value,
        'significant': significant,
        'n01': int(n01),
        'n10': int(n10),
        'winner': winner,
        'accuracy_1': acc1,
        'accuracy_2': acc2
    }

    logger.info(f"\nMcNemar's test: {model1_name} vs {model2_name}")
    logger.info(f"  Model1 correct only: {n10}")
    logger.info(f"  Model2 correct only: {n01}")
    logger.info(f"  Statistic: {statistic:.4f}")
    logger.info(f"  p-value: {p_value:.6f}")
    logger.info(f"  Significant: {significant}")

    return results


def friedman_test(errors_dict: Dict[str, np.ndarray]) -> Dict:
    """
    Friedman test for comparing multiple models.

    Non-parametric test for >2 models.

    Parameters
    ----------
    errors_dict : dict
        Dictionary {model_name: errors_array}

    Returns
    -------
    dict
        Test results
    """
    model_names = list(errors_dict.keys())
    errors_arrays = [np.abs(errors_dict[name]) for name in model_names]

    # Friedman test
    statistic, p_value = stats.friedmanchisquare(*errors_arrays)

    significant = p_value < 0.05

    # Rank models by mean error
    mean_errors = {name: np.mean(np.abs(errors_dict[name]))
                   for name in model_names}
    ranked = sorted(mean_errors.items(), key=lambda x: x[1])

    results = {
        'statistic': statistic,
        'p_value': p_value,
        'significant': significant,
        'ranking': ranked
    }

    logger.info(f"\nFriedman test (comparing {len(model_names)} models)")
    logger.info(f"  Statistic: {statistic:.4f}")
    logger.info(f"  p-value: {p_value:.6f}")

    if significant:
        logger.info(f"  ✓ At least one model is significantly different")
        logger.info(f"\n  Ranking (best to worst):")
        for rank, (name, error) in enumerate(ranked, 1):
            logger.info(f"    {rank}. {name}: {error:.4f}")
    else:
        logger.info(f"  ✗ No significant differences among models")

    return results


def compare_all_models_regression(y_true: np.ndarray,
                                  predictions_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Pairwise statistical comparison of all regression models.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    predictions_dict : dict
        Dictionary {model_name: predictions}

    Returns
    -------
    pd.DataFrame
        Matrix of p-values for all pairwise comparisons
    """
    logger.info("\n" + "="*60)
    logger.info("PAIRWISE MODEL COMPARISON (Regression)")
    logger.info("="*60)

    model_names = list(predictions_dict.keys())
    n_models = len(model_names)

    # Compute errors
    errors_dict = {name: y_true - predictions_dict[name]
                   for name in model_names}

    # Pairwise t-tests
    results_matrix = pd.DataFrame(index=model_names, columns=model_names)

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:  # Only upper triangle
                result = paired_t_test(errors_dict[model1], errors_dict[model2],
                                      model1, model2)
                results_matrix.loc[model1, model2] = result['p_value']
                results_matrix.loc[model2, model1] = result['p_value']
            elif i == j:
                results_matrix.loc[model1, model2] = 1.0  # Same model

    # Friedman test (omnibus test)
    friedman_result = friedman_test(errors_dict)

    logger.info("\n" + "="*60)
    logger.info("P-VALUE MATRIX (Paired t-test)")
    logger.info("="*60)
    logger.info("\n" + results_matrix.to_string())
    logger.info("\nNote: p < 0.05 = statistically significant difference")

    return results_matrix


def compare_all_models_classification(y_true: np.ndarray,
                                      predictions_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Pairwise statistical comparison of all classification models.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    predictions_dict : dict
        Dictionary {model_name: predictions}

    Returns
    -------
    pd.DataFrame
        Matrix of p-values for all pairwise comparisons
    """
    logger.info("\n" + "="*60)
    logger.info("PAIRWISE MODEL COMPARISON (Classification)")
    logger.info("="*60)

    model_names = list(predictions_dict.keys())
    n_models = len(model_names)

    # Pairwise McNemar tests
    results_matrix = pd.DataFrame(index=model_names, columns=model_names)

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:  # Only upper triangle
                result = mcnemar_test(y_true,
                                     predictions_dict[model1],
                                     predictions_dict[model2],
                                     model1, model2)
                results_matrix.loc[model1, model2] = result['p_value']
                results_matrix.loc[model2, model1] = result['p_value']
            elif i == j:
                results_matrix.loc[model1, model2] = 1.0  # Same model

    logger.info("\n" + "="*60)
    logger.info("P-VALUE MATRIX (McNemar's test)")
    logger.info("="*60)
    logger.info("\n" + results_matrix.to_string())
    logger.info("\nNote: p < 0.05 = statistically significant difference")

    return results_matrix


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Test with dummy data
    logger.info("Testing statistical tests module...")

    np.random.seed(42)

    # Simulate test data
    n = 100
    y_true = np.random.randn(n) * 10 + 15

    # Model 1: slightly better
    pred1 = y_true + np.random.randn(n) * 2.0

    # Model 2: slightly worse
    pred2 = y_true + np.random.randn(n) * 2.5

    # Test paired t-test
    errors1 = y_true - pred1
    errors2 = y_true - pred2

    result = paired_t_test(errors1, errors2, "Model1", "Model2")

    # Test with multiple models
    pred3 = y_true + np.random.randn(n) * 3.0
    pred4 = y_true + np.random.randn(n) * 2.2

    predictions_dict = {
        'model1': pred1,
        'model2': pred2,
        'model3': pred3,
        'model4': pred4
    }

    p_matrix = compare_all_models_regression(y_true, predictions_dict)

    # Test classification
    y_true_clf = np.random.randint(0, 2, n)
    pred1_clf = (y_true_clf + np.random.randint(-1, 2, n)).clip(0, 1)
    pred2_clf = (y_true_clf + np.random.randint(-1, 2, n)).clip(0, 1)

    mcnemar_result = mcnemar_test(y_true_clf, pred1_clf, pred2_clf,
                                  "Classifier1", "Classifier2")

    logger.info("\n✓ Statistical tests module test complete!")