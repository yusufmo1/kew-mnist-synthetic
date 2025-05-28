"""Simplified utility functions for Kew-MNIST project."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support
)
import seaborn as sns


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary.
        save_path: Path to save configuration.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        if save_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif save_path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {save_path.suffix}")


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """Plot training history curves.
    
    Args:
        history: Training history dictionary.
        save_path: Optional path to save the plot.
        show: Whether to display the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Train')
    if 'val_accuracy' in history:
        ax1.plot(history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history['loss'], label='Train')
    if 'val_loss' in history:
        ax2.plot(history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    normalize: bool = False
) -> np.ndarray:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Optional path to save the plot.
        show: Whether to display the plot.
        normalize: Whether to normalize the confusion matrix.
        
    Returns:
        Confusion matrix array.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return cm


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Optional path to save metrics.
        
    Returns:
        Dictionary containing various metrics.
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Create metrics dictionary
    metrics = {
        'overall_accuracy': float(accuracy),
        'macro_avg': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1': float(macro_f1)
        },
        'weighted_avg': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1': float(weighted_f1)
        },
        'per_class': {}
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    # Print report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save metrics if requested
    if save_path:
        save_path = Path(save_path)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {save_path}")
    
    return metrics


def visualize_predictions(
    model,
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    num_samples: int = 16,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """Visualize model predictions on sample images.
    
    Args:
        model: Trained model.
        images: Test images.
        labels: True labels.
        class_names: List of class names.
        num_samples: Number of samples to visualize.
        save_path: Optional path to save the plot.
        show: Whether to display the plot.
    """
    # Ensure images have channel dimension
    if len(images.shape) == 3:
        images_with_channel = np.expand_dims(images, axis=-1)
    else:
        images_with_channel = images
    
    # Get predictions
    predictions = model.predict(images_with_channel[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Create plot
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Display image
        if len(images.shape) == 3:
            ax.imshow(images[i], cmap='gray')
        else:
            ax.imshow(images[i, :, :, 0], cmap='gray')
        
        # Add title with prediction
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted_classes[i]]
        confidence = predictions[i, predicted_classes[i]]
        
        color = 'green' if predicted_classes[i] == labels[i] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                    color=color, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved predictions visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compare_models(
    metrics_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """Compare metrics from multiple models.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics.
        save_path: Optional path to save the plot.
        show: Whether to display the plot.
    """
    model_names = list(metrics_dict.keys())
    class_names = list(next(iter(metrics_dict.values()))['per_class'].keys())
    
    # Extract per-class accuracies
    accuracies = {}
    for class_name in class_names:
        accuracies[class_name] = [
            metrics_dict[model]['per_class'][class_name]['f1']
            for model in model_names
        ]
    
    # Create bar plot
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model_name in enumerate(model_names):
        values = [accuracies[class_name][i] for class_name in class_names]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Model Comparison by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()