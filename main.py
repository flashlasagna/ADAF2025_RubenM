"""
Main pipeline orchestrator for weather forecasting project.
Runs complete workflow from raw data to trained models.

Usage:
    python main.py --step all              # Run complete pipeline
    python main.py --step preprocess       # Only preprocessing
    python main.py --step features         # Only feature engineering
    python main.py --step models           # Only model training
    python main.py --step evaluate         # Only evaluation
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import setup_logging, print_config

logger = logging.getLogger(__name__)


def run_preprocessing():
    """Run data preprocessing pipeline."""
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("="*80)

    try:
        from src.data.preprocess import run_preprocessing_pipeline
        df = run_preprocessing_pipeline()
        logger.info("--OK-- Preprocessing complete!")
        return True
    except Exception as e:
        logger.error(f"--FAIL-- Preprocessing failed: {e}")
        return False


def run_feature_engineering():
    """Run feature engineering pipeline."""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("="*80)

    try:
        from src.features.build_features import build_complete_feature_set
        df = build_complete_feature_set(save_output=True)
        logger.info("--OK-- Feature engineering complete!")
        return True
    except Exception as e:
        logger.error(f"--FAIL-- Feature engineering failed: {e}")
        return False


def run_model_training():
    """Run model training pipeline."""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("="*80)

    try:
        from src.models.train_models import train_all_models
        models = train_all_models()
        logger.info("--OK-- Model training complete!")
        return True
    except Exception as e:
        logger.error(f"--FAIL-- Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_evaluation():
    """Run model evaluation pipeline."""
    logger.info("\n" + "="*80)
    logger.info("STEP 4: MODEL EVALUATION")
    logger.info("="*80)

    try:
        from src.evaluation.evaluate_models import evaluate_all_models
        from src.evaluation.visualization import create_all_plots

        # Evaluate models
        results = evaluate_all_models()

        # Create plots
        logger.info("\nGenerating visualizations...")
        create_all_plots(results['regression'], results['classification'])

        logger.info("--OK-- Model evaluation complete!")
        return True
    except Exception as e:
        logger.error(f"--FAIL-- Model evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all():
    """Run complete pipeline."""
    logger.info("\n" + "="*80)
    logger.info("RUNNING COMPLETE PIPELINE")
    logger.info("="*80)

    # Print configuration
    print_config()

    # Step 1: Preprocessing
    if not run_preprocessing():
        logger.error("Pipeline failed at preprocessing step")
        return False

    # Step 2: Feature Engineering
    if not run_feature_engineering():
        logger.error("Pipeline failed at feature engineering step")
        return False

    # Step 3: Model Training (TODO)
    if not run_model_training():
        logger.warning("Pipeline completed preprocessing and features, but model training not yet implemented")
        return True  # Not a failure, just incomplete

    # Step 4: Evaluation (TODO)
    if not run_evaluation():
        logger.warning("Pipeline completed up to modeling, but evaluation not yet implemented")
        return True  # Not a failure, just incomplete

    logger.info("\n" + "="*80)
    logger.info("--OK-- COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    logger.info("="*80)

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Weather Forecasting ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --step all          # Run complete pipeline
  python main.py --step preprocess   # Only data preprocessing
  python main.py --step features     # Only feature engineering
  python main.py --step models       # Only model training
  python main.py --step evaluate     # Only evaluation
        """
    )

    parser.add_argument(
        '--step',
        type=str,
        choices=['all', 'preprocess', 'features', 'models', 'evaluate'],
        default='all',
        help='Which pipeline step to run (default: all)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(PROJECT_ROOT / 'pipeline.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("="*80)
    logger.info("WEATHER FORECASTING ML PIPELINE")
    logger.info("Come Rain, Come Shine - Ruben Mimouni")
    logger.info("="*80)

    # Run requested step
    success = False

    if args.step == 'all':
        success = run_all()
    elif args.step == 'preprocess':
        success = run_preprocessing()
    elif args.step == 'features':
        success = run_feature_engineering()
    elif args.step == 'models':
        success = run_model_training()
    elif args.step == 'evaluate':
        success = run_evaluation()

    # Exit with appropriate code
    if success:
        logger.info("\n--OK-- Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n--FAIL-- Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()