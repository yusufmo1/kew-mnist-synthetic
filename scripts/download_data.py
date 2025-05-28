#!/usr/bin/env python3
"""Download and prepare Kew-MNIST dataset with synthetic data."""

import click
from loguru import logger
from pathlib import Path

from kew_synthetic.data import KewMNISTLoader, SyntheticDataManager, ImageProcessor
from kew_synthetic.utils import get_config


@click.command()
@click.option('--data-dir', type=click.Path(), default='data',
              help='Directory to store downloaded data.')
@click.option('--force', is_flag=True,
              help='Force re-download even if data exists.')
@click.option('--no-synthetic', is_flag=True,
              help='Skip synthetic data download.')
@click.option('--process-only', is_flag=True,
              help='Only process already downloaded data.')
def main(data_dir: str, force: bool, no_synthetic: bool, process_only: bool):
    """Download and prepare Kew-MNIST dataset with synthetic data enhancement."""
    
    data_dir = Path(data_dir)
    logger.info(f"Data directory: {data_dir}")
    
    # Initialize components
    loader = KewMNISTLoader(data_dir / "processed")
    processor = ImageProcessor()
    
    if not process_only:
        # Step 1: Download original Kew-MNIST dataset
        logger.info("Step 1: Downloading Kew-MNIST dataset...")
        rdata_path = loader.download_from_azure(force=force)
        
        # Step 2: Load and process original data
        logger.info("Step 2: Loading data from RData file...")
        train_images, train_labels, test_images, test_labels = loader.load_from_rdata(rdata_path)
        
        # Step 3: Create train/test split and save
        logger.info("Step 3: Saving split data to disk...")
        split_dir = data_dir / "kew_mnist_split"
        loader.save_split_to_disk(
            train_images, train_labels,
            test_images, test_labels,
            split_dir
        )
        
        if not no_synthetic:
            # Step 4: Download synthetic data
            logger.info("Step 4: Downloading synthetic data...")
            synthetic_manager = SyntheticDataManager()
            synthetic_dir = synthetic_manager.download_synthetic_data(
                data_dir / "synthetic",
                force=force
            )
            
            # Step 5: Process synthetic images
            logger.info("Step 5: Processing synthetic images...")
            processed_dir = data_dir / "synthetic" / "processed_images"
            processor.process_directory(
                synthetic_dir,
                processed_dir,
                pattern="*.png"
            )
            
            # Step 6: Merge datasets
            logger.info("Step 6: Merging original and synthetic data...")
            merged_dir = data_dir / "kew_mnist_synthetic"
            added_counts = synthetic_manager.merge_datasets(
                split_dir,
                processed_dir,
                merged_dir
            )
            
            logger.info("Synthetic images added per class:")
            for class_name, count in added_counts.items():
                logger.info(f"  {class_name}: {count}")
            
            # Step 7: Validate synthetic data quality
            logger.info("Step 7: Validating synthetic data quality...")
            quality_metrics = synthetic_manager.validate_synthetic_quality(
                processed_dir,
                sample_size=50
            )
            
            # Step 8: Get final statistics
            logger.info("Step 8: Final dataset statistics...")
            stats = synthetic_manager.get_merge_statistics(merged_dir)
    
    else:
        # Process existing data
        logger.info("Processing existing data only...")
        
        # Check what data exists
        split_dir = data_dir / "kew_mnist_split"
        synthetic_dir = data_dir / "synthetic" / "processed_images"
        merged_dir = data_dir / "kew_mnist_synthetic"
        
        if split_dir.exists():
            logger.info(f"Found split data at {split_dir}")
            
            if synthetic_dir.exists() and not no_synthetic:
                logger.info(f"Found synthetic data at {synthetic_dir}")
                
                # Re-merge datasets
                logger.info("Re-merging datasets...")
                synthetic_manager = SyntheticDataManager()
                added_counts = synthetic_manager.merge_datasets(
                    split_dir,
                    synthetic_dir,
                    merged_dir
                )
                
                logger.info("Synthetic images added per class:")
                for class_name, count in added_counts.items():
                    logger.info(f"  {class_name}: {count}")
                
                stats = synthetic_manager.get_merge_statistics(merged_dir)
        else:
            logger.error("No data found to process. Run without --process-only first.")
            return
    
    logger.success("Data preparation complete!")
    logger.info(f"Original dataset: {data_dir / 'kew_mnist_split'}")
    if not no_synthetic:
        logger.info(f"Merged dataset: {data_dir / 'kew_mnist_synthetic'}")


if __name__ == '__main__':
    main()