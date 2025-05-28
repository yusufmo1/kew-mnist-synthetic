#!/usr/bin/env python3
"""Generate all analysis results and plots for the Kew-MNIST synthetic data project."""

import click
from loguru import logger
from pathlib import Path
import json
import subprocess
import sys
from datetime import datetime

def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{description}")
    logger.info(f"{'='*60}")
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        return False


@click.command()
@click.option('--data-dir', type=click.Path(exists=True), default='data',
              help='Directory containing prepared data.')
@click.option('--model-dir', type=click.Path(exists=True), default='models',
              help='Directory containing trained models.')
@click.option('--output-dir', type=click.Path(), default='results',
              help='Directory to save all analysis results.')
@click.option('--skip-download', is_flag=True,
              help='Skip data download step.')
@click.option('--skip-training', is_flag=True,
              help='Skip model training step.')
@click.option('--skip-evaluation', is_flag=True,
              help='Skip model evaluation step.')
@click.option('--skip-notebooks', is_flag=True,
              help='Skip running analysis notebooks.')
@click.option('--force', is_flag=True,
              help='Force re-run all steps even if outputs exist.')
def main(
    data_dir: str,
    model_dir: str,
    output_dir: str,
    skip_download: bool,
    skip_training: bool,
    skip_evaluation: bool,
    skip_notebooks: bool,
    force: bool
):
    """Run complete analysis pipeline for Kew-MNIST synthetic data project."""
    
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # Setup logging to file
    log_file = output_dir / "logs" / f"analysis_{timestamp}.log"
    logger.add(log_file, rotation="500 MB")
    
    logger.info("Starting complete analysis pipeline")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Track pipeline status
    pipeline_status = {
        'timestamp': timestamp,
        'steps': {}
    }
    
    # Step 1: Download and prepare data
    if not skip_download:
        if force or not (data_dir / "kew_mnist_synthetic").exists():
            success = run_command(
                [sys.executable, "scripts/download_data.py", 
                 "--data-dir", str(data_dir),
                 "--force" if force else ""],
                "Step 1: Downloading and preparing data"
            )
            pipeline_status['steps']['download'] = 'success' if success else 'failed'
            if not success:
                logger.error("Data download failed. Exiting.")
                return
        else:
            logger.info("Skipping download - data already exists")
            pipeline_status['steps']['download'] = 'skipped'
    
    # Step 2: Train models
    if not skip_training:
        if force or not list(model_dir.glob("*/final_model.keras")):
            success = run_command(
                [sys.executable, "scripts/train_models.py",
                 "--data-dir", str(data_dir),
                 "--output-dir", str(model_dir)],
                "Step 2: Training models"
            )
            pipeline_status['steps']['training'] = 'success' if success else 'failed'
            if not success:
                logger.error("Model training failed. Exiting.")
                return
        else:
            logger.info("Skipping training - models already exist")
            pipeline_status['steps']['training'] = 'skipped'
    
    # Step 3: Evaluate models
    if not skip_evaluation:
        eval_output = output_dir / "evaluation"
        success = run_command(
            [sys.executable, "scripts/evaluate.py",
             "--model-dir", str(model_dir),
             "--data-dir", str(data_dir),
             "--output-dir", str(eval_output)],
            "Step 3: Evaluating models"
        )
        pipeline_status['steps']['evaluation'] = 'success' if success else 'failed'
        if not success:
            logger.error("Model evaluation failed. Continuing...")
    
    # Step 4: Run analysis notebooks
    if not skip_notebooks:
        logger.info(f"\n{'='*60}")
        logger.info("Step 4: Running analysis notebooks")
        logger.info(f"{'='*60}")
        
        notebooks = [
            "notebooks/01_data_exploration.ipynb",
            "notebooks/02_model_comparison.ipynb",
            "notebooks/03_results_analysis.ipynb"
        ]
        
        for notebook in notebooks:
            if Path(notebook).exists():
                logger.info(f"Running {notebook}...")
                success = run_command(
                    ["jupyter", "nbconvert", "--to", "html",
                     "--execute", notebook,
                     "--output-dir", str(output_dir / "notebooks"),
                     "--ExecutePreprocessor.timeout=600"],
                    f"Executing {notebook}"
                )
                pipeline_status['steps'][f'notebook_{Path(notebook).stem}'] = \
                    'success' if success else 'failed'
            else:
                logger.warning(f"Notebook {notebook} not found")
                pipeline_status['steps'][f'notebook_{Path(notebook).stem}'] = 'not_found'
    
    # Step 5: Generate final report
    logger.info(f"\n{'='*60}")
    logger.info("Step 5: Generating final report")
    logger.info(f"{'='*60}")
    
    # Collect all results
    final_report = {
        'pipeline_status': pipeline_status,
        'data_statistics': {},
        'model_performance': {},
        'improvement_summary': {}
    }
    
    # Load data statistics if available
    merge_stats_path = data_dir / "kew_mnist_synthetic" / "merge_statistics.json"
    if merge_stats_path.exists():
        with open(merge_stats_path) as f:
            final_report['data_statistics'] = json.load(f)
    
    # Load evaluation results if available
    eval_results_path = output_dir / "evaluation" / "evaluation_results.json"
    if eval_results_path.exists():
        with open(eval_results_path) as f:
            eval_data = json.load(f)
            final_report['model_performance'] = eval_data.get('comparison', {})
            final_report['improvement_summary'] = eval_data.get('improvement', {})
    
    # Save final report
    report_path = output_dir / f"final_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*60)
    
    logger.info("\nPipeline Status:")
    for step, status in pipeline_status['steps'].items():
        logger.info(f"  {step}: {status}")
    
    if final_report['improvement_summary']:
        logger.info("\nModel Improvement Summary:")
        imp = final_report['improvement_summary']
        if 'overall_accuracy' in imp:
            logger.info(f"  Accuracy: {imp['overall_accuracy']:+.4f} "
                       f"({imp.get('percentage_improvement', {}).get('accuracy', 0):+.2f}%)")
        if 'weighted_f1' in imp:
            logger.info(f"  F1 Score: {imp['weighted_f1']:+.4f} "
                       f"({imp.get('percentage_improvement', {}).get('f1', 0):+.2f}%)")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Final report: {report_path}")
    
    # Create a summary README
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("# Kew-MNIST Synthetic Data Analysis Results\n\n")
        f.write(f"Generated on: {timestamp}\n\n")
        f.write("## Contents\n\n")
        f.write("- `evaluation/`: Model evaluation results and plots\n")
        f.write("- `notebooks/`: Executed analysis notebooks\n")
        f.write("- `logs/`: Execution logs\n")
        f.write(f"- `final_report_{timestamp}.json`: Complete analysis summary\n\n")
        
        if final_report['improvement_summary']:
            f.write("## Key Results\n\n")
            imp = final_report['improvement_summary']
            if 'overall_accuracy' in imp:
                f.write(f"- **Accuracy Improvement**: {imp['overall_accuracy']:+.4f} "
                       f"({imp.get('percentage_improvement', {}).get('accuracy', 0):+.2f}%)\n")
            if 'weighted_f1' in imp:
                f.write(f"- **F1 Score Improvement**: {imp['weighted_f1']:+.4f} "
                       f"({imp.get('percentage_improvement', {}).get('f1', 0):+.2f}%)\n")
    
    logger.success("Analysis pipeline complete!")


if __name__ == '__main__':
    main()