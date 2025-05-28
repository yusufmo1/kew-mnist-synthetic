"""Simplified model training utilities for Kew-MNIST CNN."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight


class ModelTrainer:
    """Simple trainer for Kew-MNIST CNN models."""
    
    def __init__(self, model: keras.Model, save_dir: Optional[Union[str, Path]] = None):
        """Initialize model trainer.
        
        Args:
            model: Keras model to train.
            save_dir: Directory to save training artifacts.
        """
        self.model = model
        
        # Setup save directory
        if save_dir is None:
            save_dir = Path("models") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = None
        
        # Class names for reference
        self.class_names = ['flower', 'fruit', 'leaf', 'plant_tag', 'stem', 'whole_plant']
    
    def calculate_class_weights(self, labels: np.ndarray, method: str = "custom") -> Dict[int, float]:
        """Calculate class weights for handling imbalance.
        
        Args:
            labels: Training labels.
            method: Method for calculating weights ('balanced', 'custom', 'none').
            
        Returns:
            Dictionary mapping class indices to weights.
        """
        if method == "none":
            return None
        
        if method == "balanced":
            # Use sklearn's balanced weights
            classes = np.unique(labels)
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=labels
            )
            class_weights = {i: w for i, w in enumerate(weights)}
            
        elif method == "custom":
            # Custom weights based on the original notebook
            class_weights = {
                0: 2.1,    # flower
                1: 0.74,   # fruit
                2: 0.36,   # leaf
                3: 2.5,    # plant_tag
                4: 1.7,    # stem
                5: 0.64    # whole_plant
            }
        else:
            raise ValueError(f"Unknown class weight method: {method}")
        
        print("Class weights:")
        for idx, weight in class_weights.items():
            print(f"  {self.class_names[idx]}: {weight:.3f}")
        
        return class_weights
    
    def setup_callbacks(self) -> List[keras.callbacks.Callback]:
        """Setup standard training callbacks.
        
        Returns:
            List of callback instances.
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = self.save_dir / "checkpoints" / "best_model.keras"
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ))
        
        # Early stopping
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            mode='min',
            verbose=1
        ))
        
        # Reduce learning rate on plateau
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='min',
            verbose=1
        ))
        
        # CSV logger
        csv_path = self.save_dir / "training_log.csv"
        callbacks.append(keras.callbacks.CSVLogger(
            str(csv_path),
            append=False
        ))
        
        return callbacks
    
    def train(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 35,
        batch_size: int = 32,
        class_weight_method: str = "custom",
        callbacks: Optional[List] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """Train the model.
        
        Args:
            train_images: Training images.
            train_labels: Training labels.
            validation_data: Optional validation (images, labels).
            epochs: Number of epochs.
            batch_size: Batch size.
            class_weight_method: Method for class weights.
            callbacks: Additional callbacks. If None, uses setup_callbacks().
            verbose: Verbosity level.
            
        Returns:
            Training history object.
        """
        # Add channel dimension if needed
        if len(train_images.shape) == 3:
            train_images = np.expand_dims(train_images, axis=-1)
        if validation_data is not None and len(validation_data[0].shape) == 3:
            validation_data = (
                np.expand_dims(validation_data[0], axis=-1),
                validation_data[1]
            )
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_labels, class_weight_method)
        
        # Setup callbacks
        if callbacks is None:
            callbacks = self.setup_callbacks()
        
        # Log training info
        print(f"Starting training for {epochs} epochs with batch size {batch_size}")
        print(f"Training samples: {len(train_images)}")
        if validation_data:
            print(f"Validation samples: {len(validation_data[0])}")
        
        # Train model
        self.history = self.model.fit(
            train_images,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        # Save final model
        self.save_model("final_model.keras")
        
        # Save training history
        self.save_training_history()
        
        return self.history
    
    def save_model(self, filename: str = "model.keras") -> Path:
        """Save trained model.
        
        Args:
            filename: Model filename.
            
        Returns:
            Path to saved model.
        """
        model_path = self.save_dir / filename
        self.model.save(str(model_path))
        print(f"Saved model to {model_path}")
        
        # Save metadata
        metadata = {
            "filename": filename,
            "saved_date": datetime.now().isoformat(),
            "model_name": self.model.name,
            "parameters": self.model.count_params(),
            "input_shape": list(self.model.input_shape[1:]),
            "num_classes": self.model.output_shape[-1]
        }
        
        # Add performance metrics if available
        if self.history:
            final_epoch = len(self.history.history['loss']) - 1
            metadata["final_metrics"] = {
                "loss": float(self.history.history['loss'][final_epoch]),
                "accuracy": float(self.history.history['accuracy'][final_epoch])
            }
            
            if 'val_loss' in self.history.history:
                metadata["final_metrics"]["val_loss"] = float(
                    self.history.history['val_loss'][final_epoch]
                )
                metadata["final_metrics"]["val_accuracy"] = float(
                    self.history.history['val_accuracy'][final_epoch]
                )
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def save_training_history(self) -> Path:
        """Save training history to JSON file.
        
        Returns:
            Path to saved history file.
        """
        if self.history is None:
            print("No training history to save")
            return None
        
        history_path = self.save_dir / "training_history.json"
        
        # Convert numpy values to Python types
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        # Add training configuration
        history_data = {
            "history": history_dict,
            "params": self.history.params,
            "model_name": self.model.name,
            "total_params": self.model.count_params()
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Saved training history to {history_path}")
        return history_path
    
    def evaluate(
        self,
        test_images: np.ndarray,
        test_labels: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            test_images: Test images.
            test_labels: Test labels.
            batch_size: Batch size for evaluation.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Add channel dimension if needed
        if len(test_images.shape) == 3:
            test_images = np.expand_dims(test_images, axis=-1)
        
        # Evaluate
        results = self.model.evaluate(
            test_images,
            test_labels,
            batch_size=batch_size,
            verbose=1
        )
        
        # Create metrics dictionary
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = float(results[i])
        
        print("Evaluation results:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        
        return metrics
    
    @staticmethod
    def load_model(model_path: Union[str, Path]) -> keras.Model:
        """Load a saved model.
        
        Args:
            model_path: Path to saved model.
            
        Returns:
            Loaded Keras model.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = keras.models.load_model(str(model_path))
        print(f"Loaded model from {model_path}")
        
        return model