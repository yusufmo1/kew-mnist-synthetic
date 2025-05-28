#!/usr/bin/env python3
"""Evaluate and compare trained Kew-MNIST models."""

import click
from loguru import logger
from pathlib import Path
import json

from kew_synthetic.data import KewMNISTLoader
from kew_synthetic.evaluation import ModelEvaluator, OcclusionAnalyzer, ResultVisualizer
from kew_synthetic.models import ModelTrainer


@click.command()
@click.option('--model-dir', type=click.Path(exists=True), required=True,
              help='Directory containing trained models.')
@click.option('--data-dir', type=click.Path(exists=True), default='data',
              help='Directory containing test data.')
@click.option('--output-dir', type=click.Path(), default='results',
              help='Directory to save evaluation results.')
@click.option('--no-occlusion', is_flag=True,
              help='Skip occlusion sensitivity analysis.')
@click.option('--no-plots', is_flag=True,
              help='Skip generating plots.')
@click.option('--original-model', type=click.Path(exists=True),
              help='Path to original model file.')
@click.option('--synthetic-model', type=click.Path(exists=True),
              help='Path to synthetic model file.')
def main(
    model_dir: str,
    data_dir: str,
    output_dir: str,
    no_occlusion: bool,
    no_plots: bool,
    original_model: str,
    synthetic_model: str
):
    """Comprehensive evaluation of trained models."""
    
    model_dir = Path(model_dir)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    loader = KewMNISTLoader()
    evaluator = ModelEvaluator()
    
    # Find model files
    if original_model and synthetic_model:
        original_path = Path(original_model)
        synthetic_path = Path(synthetic_model)
    else:
        # Auto-detect models
        original_paths = list(model_dir.glob("original*/final_model.keras"))
        synthetic_paths = list(model_dir.glob("synthetic*/final_model.keras"))
        
        if not original_paths or not synthetic_paths:
            logger.error("Could not find both original and synthetic models")
            logger.info("Specify paths with --original-model and --synthetic-model")
            return
        
        # Use most recent
        original_path = sorted(original_paths)[-1]
        synthetic_path = sorted(synthetic_paths)[-1]
    
    logger.info(f"Original model: {original_path}")
    logger.info(f"Synthetic model: {synthetic_path}")
    
    # Load models
    original_model = ModelTrainer.load_model(original_path)
    synthetic_model = ModelTrainer.load_model(synthetic_path)
    
    # Load test data (use original test set for fair comparison)
    test_dir = data_dir / "kew_mnist_split"
    if not test_dir.exists():
        logger.error(f"Test data not found at {test_dir}")
        return
    
    test_images, test_labels = loader.load_from_directory(test_dir, "test")
    
    # Evaluate models
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATING MODELS")
    logger.info("=" * 50)
    
    models = [
        (original_model, "Original"),
        (synthetic_model, "Synthetic-Enhanced")
    ]
    
    # Comprehensive evaluation
    comparison = evaluator.compare_models(models, test_images, test_labels)
    
    # Calculate improvements
    improvement = evaluator.calculate_improvement("Original", "Synthetic-Enhanced")
    
    # Save evaluation results
    results = {
        'model_paths': {
            'original': str(original_path),
            'synthetic': str(synthetic_path)
        },
        'comparison': comparison,
        'improvement': improvement,
        'individual_results': evaluator.results
    }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nSaved evaluation results to {results_path}")
    
    # Generate visualizations
    if not no_plots:
        logger.info("\n" + "=" * 50)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 50)
        
        visualizer = ResultVisualizer(save_dir=output_dir / "plots")
        
        # Plot confusion matrices
        for name, result in evaluator.results.items():
            visualizer.plot_confusion_matrix(
                result['confusion_matrix'],
                evaluator.class_names,
                title=f"{name} Model - Confusion Matrix"
            )
        
        # Plot ROC curves
        for name, result in evaluator.results.items():
            visualizer.plot_roc_curves(
                result['roc_auc']['curves']['fpr'],
                result['roc_auc']['curves']['tpr'],
                result['roc_auc']['per_class'],
                evaluator.class_names,
                title=f"{name} Model - ROC Curves"
            )
        
        # Plot per-class accuracy comparison
        accuracies = {
            name: result['accuracy']['per_class']
            for name, result in evaluator.results.items()
        }
        accuracies['Original']['overall'] = evaluator.results['Original']['accuracy']['overall']
        accuracies['Synthetic-Enhanced']['overall'] = evaluator.results['Synthetic-Enhanced']['accuracy']['overall']
        
        visualizer.plot_per_class_accuracy(
            accuracies,
            evaluator.class_names,
            title="Model Performance Comparison"
        )
        
        # Create comparison dashboard
        visualizer.create_comparison_dashboard(
            evaluator.results,
            evaluator.class_names
        )
        
        # Plot sample predictions
        for name, result in evaluator.results.items():
            visualizer.plot_sample_predictions(
                test_images,
                test_labels,
                result['predictions'],
                result['probabilities'],
                evaluator.class_names,
                num_samples=15,
                title=f"{name} Model - Sample Predictions"
            )
    
    # Occlusion sensitivity analysis
    if not no_occlusion:
        logger.info("\n" + "=" * 50)
        logger.info("OCCLUSION SENSITIVITY ANALYSIS")
        logger.info("=" * 50)
        
        analyzer = OcclusionAnalyzer()
        
        # Find model differences
        occlusion_results = analyzer.find_model_differences(
            original_model,
            synthetic_model,
            test_images,
            test_labels,
            evaluator.class_names,
            examples_per_class=1
        )
        
        # Visualize results
        if occlusion_results:
            analyzer.visualize_heatmaps(
                save_dir=output_dir / "occlusion"
            )
            
            analyzer.create_summary_visualization(
                save_path=output_dir / "occlusion" / "summary.png"
            )
        else:
            logger.warning("No suitable examples found for occlusion analysis")
    
    # Print final summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    
    logger.info(f"\nOverall Accuracy:")
    logger.info(f"  Original: {comparison['overall_accuracy']['Original']:.4f}")
    logger.info(f"  Synthetic: {comparison['overall_accuracy']['Synthetic-Enhanced']:.4f}")
    logger.info(f"  Improvement: {improvement['overall_accuracy']:+.4f} "
               f"({improvement['percentage_improvement']['accuracy']:+.2f}%)")
    
    logger.info(f"\nWeighted F1 Score:")
    logger.info(f"  Original: {comparison['weighted_f1']['Original']:.4f}")
    logger.info(f"  Synthetic: {comparison['weighted_f1']['Synthetic-Enhanced']:.4f}")
    logger.info(f"  Improvement: {improvement['weighted_f1']:+.4f} "
               f"({improvement['percentage_improvement']['f1']:+.2f}%)")
    
    logger.info("\nPer-class Accuracy Improvements:")
    for class_name, imp in improvement['per_class_accuracy'].items():
        logger.info(f"  {class_name}: {imp:+.4f}")
    
    logger.success("\nEvaluation complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()