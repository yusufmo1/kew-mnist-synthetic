"""Simplified data loading utilities for Kew-MNIST dataset."""

import os
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed


class DataLoader:
    """Simple data loader for Kew-MNIST dataset."""
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """Initialize data loader.
        
        Args:
            data_dir: Root directory for data storage.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Class names mapping
        self.class_names = ['flower', 'fruit', 'leaf', 'plant_tag', 'stem', 'whole_plant']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
    
    def download_from_azure(self, force: bool = False) -> Path:
        """Download Kew-MNIST dataset from Azure blob storage.
        
        Args:
            force: Force re-download even if file exists.
            
        Returns:
            Path to downloaded RData file.
        """
        raw_data_dir = self.data_dir / "raw"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        rdata_path = raw_data_dir / "Kew-MNIST-full-dataset.Rdata"
        
        # Check if already downloaded
        if rdata_path.exists() and not force:
            print(f"Dataset already exists at {rdata_path}")
            return rdata_path
        
        # Download from Azure
        url = "https://kewmniststorage.blob.core.windows.net/kew-mnist/Kew-MNIST-full-dataset.Rdata"
        print(f"Downloading Kew-MNIST dataset from {url}")
        
        try:
            cmd = ["wget", "--progress=bar:force", "-O", str(rdata_path), url]
            subprocess.run(cmd, check=True)
            print(f"Downloaded dataset to {rdata_path}")
            return rdata_path
        except subprocess.CalledProcessError as e:
            print(f"Failed to download dataset: {e}")
            raise
    
    def load_from_directory(
        self, 
        data_dir: Union[str, Path], 
        split: str = "train",
        n_jobs: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load processed images from directory structure.
        
        Args:
            data_dir: Root directory containing train/test subdirectories.
            split: Which split to load ("train" or "test").
            n_jobs: Number of parallel jobs. -1 means use all CPUs minus 1.
            
        Returns:
            Tuple of (images, labels) arrays.
        """
        data_dir = Path(data_dir)
        split_dir = data_dir / split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Read labels CSV
        labels_csv = data_dir / f"{split}_labels.csv"
        if not labels_csv.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_csv}")
        
        df = pd.read_csv(labels_csv)
        n_images = len(df)
        
        print(f"Loading {n_images} {split} images from {split_dir}")
        
        # Prepare image loading parameters
        img_info_list = [
            (i, row['filename'], row['class_index'], row['class_name'], split_dir)
            for i, row in df.iterrows()
        ]
        
        # Determine number of jobs
        if n_jobs == -1:
            n_jobs = max(1, os.cpu_count() - 1)
        
        # Load images in parallel
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._load_single_image)(img_info)
            for img_info in tqdm(img_info_list, desc=f"Loading {split} images")
        )
        
        # Allocate arrays
        images = np.zeros((n_images, 500, 500), dtype=np.float64)
        labels = np.zeros(n_images, dtype=np.int64)
        
        # Fill arrays
        for idx, img_array, class_idx in results:
            images[idx] = img_array
            labels[idx] = class_idx
        
        print(f"Loaded {n_images} {split} images")
        
        # Print class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  {self.idx_to_class[label]}: {count} images")
        
        return images, labels
    
    def _load_single_image(self, img_info: Tuple) -> Tuple[int, np.ndarray, int]:
        """Load a single image from disk.
        
        Args:
            img_info: Tuple of (index, filename, class_index, class_name, base_dir).
            
        Returns:
            Tuple of (index, image_array, class_index).
        """
        idx, filename, class_index, class_name, base_dir = img_info
        img_path = base_dir / class_name / filename
        
        try:
            # Load image
            pil_image = Image.open(img_path)
            img_array = np.array(pil_image)
            
            # Handle grayscale vs RGB
            if len(img_array.shape) == 3:
                img_array = img_array[:, :, 0]  # Take first channel
            
            # Validate shape
            if img_array.shape != (500, 500):
                raise ValueError(f"Image {filename} has shape {img_array.shape}, expected (500, 500)")
            
            return (idx, img_array, int(class_index))
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return zeros array as fallback
            return (idx, np.zeros((500, 500), dtype=np.float64), int(class_index))
    
    def create_train_test_split(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.3,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create train/test split with stratification.
        
        Args:
            images: Array of images.
            labels: Array of labels.
            test_size: Fraction of data for test set.
            random_state: Random seed.
            stratify: Whether to stratify by class labels.
            
        Returns:
            Tuple of (train_images, train_labels, test_images, test_labels).
        """
        print(f"Creating train/test split with test_size={test_size}, random_state={random_state}")
        
        # Perform split
        stratify_labels = labels if stratify else None
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        # Log statistics
        print(f"Training set: {len(train_images)} images")
        print(f"Test set: {len(test_images)} images")
        
        # Log class distribution
        for split_name, split_labels in [("train", train_labels), ("test", test_labels)]:
            print(f"\n{split_name.capitalize()} class distribution:")
            for i, class_name in enumerate(self.class_names):
                count = np.sum(split_labels == i)
                percentage = count / len(split_labels) * 100
                print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        return train_images, train_labels, test_images, test_labels
    
    def merge_synthetic_data(
        self,
        original_images: np.ndarray,
        original_labels: np.ndarray,
        synthetic_dir: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge original data with synthetic data.
        
        Args:
            original_images: Original training images.
            original_labels: Original training labels.
            synthetic_dir: Directory containing synthetic images.
            
        Returns:
            Tuple of (merged_images, merged_labels).
        """
        synthetic_dir = Path(synthetic_dir)
        if not synthetic_dir.exists():
            raise FileNotFoundError(f"Synthetic directory not found: {synthetic_dir}")
        
        print(f"Loading synthetic data from {synthetic_dir}")
        
        # Load synthetic images
        synthetic_images = []
        synthetic_labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = synthetic_dir / class_name
            if not class_dir.exists():
                print(f"Warning: No synthetic data for class {class_name}")
                continue
            
            # Load all images from this class
            for img_path in sorted(class_dir.glob("*.png")):
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize((500, 500), Image.LANCZOS)
                    img_array = np.array(img)
                    
                    synthetic_images.append(img_array)
                    synthetic_labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading synthetic image {img_path}: {e}")
        
        if synthetic_images:
            synthetic_images = np.array(synthetic_images)
            synthetic_labels = np.array(synthetic_labels)
            
            print(f"Loaded {len(synthetic_images)} synthetic images")
            
            # Merge with original data
            merged_images = np.concatenate([original_images, synthetic_images], axis=0)
            merged_labels = np.concatenate([original_labels, synthetic_labels], axis=0)
            
            print(f"Total merged dataset: {len(merged_images)} images")
            
            return merged_images, merged_labels
        else:
            print("No synthetic images loaded, returning original data")
            return original_images, original_labels