#!/usr/bin/env python3
"""Train Kew-MNIST models with and without synthetic data."""

import click
from datetime import datetime
from loguru import logger
from pathlib import Path
import json

from kew_synthetic.data import KewMNISTLoader
from kew_synthetic.models import create_kew_cnn, ModelTrainer
from kew_synthetic.utils import get_config


@click.command()
@click.option('--data-dir', type=click.Path(exists=True), default='data',
              help='Directory containing prepared data.')
@click.option('--output-dir', type=click.Path(), default='models',
              help='Directory to save trained models.')
@click.option('--original-only', is_flag=True,
              help='Train only on original data.')
@click.option('--synthetic-only', is_flag=True,
              help='Train only on synthetic-enhanced data.')
@click.option('--epochs', type=int, default=None,
              help='Number of training epochs (overrides config).')
@click.option('--batch-size', type=int, default=None,
              help='Batch size (overrides config).')
@click.option('--no-class-weights', is_flag=True,
              help='Disable class weights.')
def main(
    data_dir: str,
    output_dir: str,
    original_only: bool,
    synthetic_only: bool,
    epochs: int,
    batch_size: int,
    no_class_weights: bool
):
    """Train Kew-MNIST CNN models with comparison."""
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize data loader
    loader = KewMNISTLoader()
    
    # Determine which models to train
    train_original = not synthetic_only
    train_synthetic = not original_only
    
    results = {}
    
    # Train original model
    if train_original:
        logger.info("=" * 50)
        logger.info("Training on original Kew-MNIST dataset")
        logger.info("=" * 50)
        
        # Load original data
        original_dir = data_dir / "kew_mnist_split"
        if not original_dir.exists():
            logger.error(f"Original data not found at {original_dir}")
            logger.info("Run 'python scripts/download_data.py' first")
            return
        
        train_images, train_labels = loader.load_from_directory(original_dir, "train")
        test_images, test_labels = loader.load_from_directory(original_dir, "test")
        
        # Create model and trainer
        model_original = create_kew_cnn()
        trainer_original = ModelTrainer(
            model_original,
            save_dir=output_dir / f"original_{timestamp}"
        )
        
        # Train model
        history_original = trainer_original.train(
            train_images,
            train_labels,
            validation_data=(test_images, test_labels),
            epochs=epochs,
            batch_size=batch_size,
            class_weight_method="none" if no_class_weights else "custom"
        )
        
        # Evaluate
        metrics_original = trainer_original.evaluate(test_images, test_labels)
        results['original'] = {
            'save_dir': str(trainer_original.save_dir),
            'final_metrics': metrics_original,
            'epochs_trained': len(history_original.history['loss'])
        }
    
    # Train synthetic-enhanced model
    if train_synthetic:
        logger.info("\n" + "=" * 50)
        logger.info("Training on synthetic-enhanced dataset")
        logger.info("=" * 50)
        
        # Load synthetic-enhanced data
        synthetic_dir = data_dir / "kew_mnist_synthetic"
        if not synthetic_dir.exists():
            logger.error(f"Synthetic data not found at {synthetic_dir}")
            logger.info("Run 'python scripts/download_data.py' first")
            return
        
        train_images_syn, train_labels_syn = loader.load_from_directory(synthetic_dir, "train")
        test_images_syn, test_labels_syn = loader.load_from_directory(synthetic_dir, "test")
        
        # Create model and trainer
        model_synthetic = create_kew_cnn()
        trainer_synthetic = ModelTrainer(
            model_synthetic,
            save_dir=output_dir / f"synthetic_{timestamp}"
        )
        
        # Train model
        history_synthetic = trainer_synthetic.train(
            train_images_syn,
            train_labels_syn,
            validation_data=(test_images_syn, test_labels_syn),
            epochs=epochs,
            batch_size=batch_size,
            class_weight_method="none" if no_class_weights else "custom"
        )
        
        # Evaluate
        metrics_synthetic = trainer_synthetic.evaluate(test_images_syn, test_labels_syn)
        results['synthetic'] = {
            'save_dir': str(trainer_synthetic.save_dir),
            'final_metrics': metrics_synthetic,
            'epochs_trained': len(history_synthetic.history['loss'])
        }
    
    # Save training summary
    if results:
        summary_path = output_dir / f"training_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 50)
        
        for model_type, info in results.items():
            logger.info(f"\n{model_type.upper()} Model:")
            logger.info(f"  Save directory: {info['save_dir']}")
            logger.info(f"  Epochs trained: {info['epochs_trained']}")
            logger.info(f"  Final metrics:")
            for metric, value in info['final_metrics'].items():
                logger.info(f"    {metric}: {value:.4f}")
        
        if len(results) == 2:
            # Compare results
            acc_diff = (results['synthetic']['final_metrics']['accuracy'] - 
                       results['original']['final_metrics']['accuracy'])
            logger.info(f"\nAccuracy improvement: {acc_diff:+.4f} "
                       f"({acc_diff*100:+.2f}%)")
    
    logger.success("Training complete!")


if __name__ == '__main__':
    main()