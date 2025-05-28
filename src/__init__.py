"""Simplified source modules for Kew-MNIST Synthetic project."""

from .data_loader import DataLoader
from .models import create_kew_cnn, create_simple_cnn
from .trainer import ModelTrainer
from .utils import (
    load_config,
    save_config,
    plot_training_history,
    plot_confusion_matrix,
    calculate_metrics,
    visualize_predictions,
    compare_models
)

__version__ = "1.0.0"

__all__ = [
    "DataLoader",
    "create_kew_cnn",
    "create_simple_cnn", 
    "ModelTrainer",
    "load_config",
    "save_config",
    "plot_training_history",
    "plot_confusion_matrix",
    "calculate_metrics",
    "visualize_predictions",
    "compare_models"
]