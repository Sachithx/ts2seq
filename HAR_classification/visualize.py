"""
VISUALIZATION AND MONITORING UTILITIES
Plot training curves, confusion matrices, and generate reports
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


def plot_training_history(results_path, save_dir=None):
    """Plot training history from results JSON."""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    history = results['history']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epoch times
    axes[1, 1].bar(epochs, history['epoch_times'], color='purple', alpha=0.6)
    axes[1, 1].axhline(y=np.mean(history['epoch_times']), color='r', 
                       linestyle='--', label=f'Mean: {np.mean(history["epoch_times"]):.1f}s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title('Training Time per Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training history plot to: {save_path}")
    
    plt.show()
    
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot confusion matrix."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = ['Walking', 'Walking Up', 'Walking Down', 'Sitting', 'Standing', 'Laying']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to: {save_path}")
    
    plt.show()
    
    return fig, cm


@torch.no_grad()
def evaluate_model(model, dataloader, device='cuda'):
    """Evaluate model and get predictions."""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc='Evaluating'):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)


def generate_classification_report(y_true, y_pred, class_names=None, save_path=None):
    """Generate and print classification report."""
    
    if class_names is None:
        class_names = ['Walking', 'Walking Up', 'Walking Down', 'Sitting', 'Standing', 'Laying']
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(report)
    print("="*80 + "\n")
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(report)
            f.write("="*80 + "\n")
        print(f"✓ Saved classification report to: {save_path}")
    
    return report


def plot_per_class_accuracy(y_true, y_pred, class_names=None, save_path=None):
    """Plot per-class accuracy."""
    
    if class_names is None:
        class_names = ['Walking', 'Walking Up', 'Walking Down', 'Sitting', 'Standing', 'Laying']
    
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(class_names, per_class_acc * 100, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc * 100):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Activity Class', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved per-class accuracy plot to: {save_path}")
    
    plt.show()
    
    return fig


def compare_experiments(result_paths, labels=None, save_dir=None):
    """Compare multiple training experiments."""
    
    if labels is None:
        labels = [f"Experiment {i+1}" for i in range(len(result_paths))]
    
    all_results = []
    for path in result_paths:
        with open(path, 'r') as f:
            all_results.append(json.load(f))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
    
    # Validation accuracy comparison
    for i, (results, label) in enumerate(zip(all_results, labels)):
        history = results['history']
        epochs = range(1, len(history['val_acc']) + 1)
        axes[0].plot(epochs, history['val_acc'], marker='o', label=label, linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('Validation Accuracy Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final test accuracy comparison
    test_accs = [results['test_acc'] for results in all_results]
    bars = axes[1].bar(labels, test_accs, color='skyblue', alpha=0.7, edgecolor='black')
    
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Final Test Accuracy')
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Training time comparison
    train_times = [results['total_time_minutes'] for results in all_results]
    bars = axes[2].bar(labels, train_times, color='coral', alpha=0.7, edgecolor='black')
    
    for bar, time in zip(bars, train_times):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}m', ha='center', va='bottom', fontweight='bold')
    
    axes[2].set_ylabel('Training Time (minutes)')
    axes[2].set_title('Total Training Time')
    axes[2].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'experiment_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved experiment comparison to: {save_path}")
    
    plt.show()
    
    return fig


def create_training_report(checkpoint_dir, output_path='training_report.md'):
    """Create a comprehensive markdown training report."""
    
    checkpoint_dir = Path(checkpoint_dir)
    results_path = checkpoint_dir / 'training_results.json'
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create report
    report = []
    report.append("# HAR Classification Training Report\n")
    report.append(f"**Checkpoint Directory:** `{checkpoint_dir}`\n")
    report.append("---\n\n")
    
    # Configuration
    report.append("## Configuration\n")
    config = results.get('config', {})
    report.append(f"- **Learning Rate:** {config.get('learning_rate', 'N/A')}")
    report.append(f"- **Epochs:** {config.get('num_epochs', 'N/A')}")
    report.append(f"- **Gradient Accumulation:** {config.get('gradient_accumulation_steps', 'N/A')}")
    report.append(f"- **Mixed Precision:** {config.get('use_amp', 'N/A')}")
    report.append(f"- **Warmup Epochs:** {config.get('warmup_epochs', 'N/A')}\n\n")
    
    # Results
    report.append("## Results\n")
    report.append(f"- **Best Validation Accuracy:** {results['best_val_acc']:.2f}%")
    report.append(f"- **Best Epoch:** {results['best_epoch'] + 1}")
    report.append(f"- **Test Accuracy:** {results['test_acc']:.2f}%")
    report.append(f"- **Test Loss:** {results['test_loss']:.4f}")
    report.append(f"- **Total Training Time:** {results['total_time_minutes']:.1f} minutes\n\n")
    
    # Training history summary
    history = results['history']
    report.append("## Training History Summary\n")
    report.append(f"- **Final Train Loss:** {history['train_loss'][-1]:.4f}")
    report.append(f"- **Final Train Accuracy:** {history['train_acc'][-1]:.2f}%")
    report.append(f"- **Final Val Loss:** {history['val_loss'][-1]:.4f}")
    report.append(f"- **Final Val Accuracy:** {history['val_acc'][-1]:.2f}%")
    report.append(f"- **Average Epoch Time:** {np.mean(history['epoch_times']):.1f} seconds\n\n")
    
    # Best epoch info
    best_idx = results['best_epoch']
    report.append(f"## Best Epoch Details (Epoch {best_idx + 1})\n")
    report.append(f"- **Train Loss:** {history['train_loss'][best_idx]:.4f}")
    report.append(f"- **Train Accuracy:** {history['train_acc'][best_idx]:.2f}%")
    report.append(f"- **Val Loss:** {history['val_loss'][best_idx]:.4f}")
    report.append(f"- **Val Accuracy:** {history['val_acc'][best_idx]:.2f}%")
    report.append(f"- **Learning Rate:** {history['learning_rates'][best_idx]:.6f}\n\n")
    
    # Files
    report.append("## Generated Files\n")
    report.append(f"- Best model: `{checkpoint_dir}/best_model.pth`")
    report.append(f"- Latest checkpoint: `{checkpoint_dir}/latest_checkpoint.pth`")
    report.append(f"- Training results: `{checkpoint_dir}/training_results.json`")
    
    if (checkpoint_dir / 'logs').exists():
        report.append(f"- TensorBoard logs: `{checkpoint_dir}/logs/`\n\n")
    else:
        report.append("\n\n")
    
    report.append("---\n")
    report.append("*Report generated automatically*\n")
    
    # Write report
    report_text = "\n".join(report)
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"✓ Training report saved to: {output_path}")
    print("\nReport Preview:")
    print("="*80)
    print(report_text)
    print("="*80)
    
    return report_text


if __name__ == '__main__':
    print("This is a utility module for visualization and monitoring.")
    print("\nAvailable functions:")
    print("  - plot_training_history()")
    print("  - plot_confusion_matrix()")
    print("  - evaluate_model()")
    print("  - generate_classification_report()")
    print("  - plot_per_class_accuracy()")
    print("  - compare_experiments()")
    print("  - create_training_report()")
