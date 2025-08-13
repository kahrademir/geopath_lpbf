import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import matplotlib as mpl

mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

def load_training_history(checkpoint_path):
    """
    Load training history from a checkpoint directory.
    
    Args:
        checkpoint_path (str): Path to the checkpoint directory
        
    Returns:
        dict: Dictionary containing training and validation losses, and config info
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load config
    config_file = checkpoint_path / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Load training losses
    train_losses_file = checkpoint_path / 'train_losses.npy'
    val_losses_file = checkpoint_path / 'val_losses.npy'
    
    train_losses = None
    val_losses = None
    
    if train_losses_file.exists():
        train_losses = np.load(train_losses_file)
    if val_losses_file.exists():
        val_losses = np.load(val_losses_file)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
        'checkpoint_name': checkpoint_path.name
    }

def plot_training_curves(histories, save_path=None, figsize=(12, 8)):
    """
    Plot training curves for multiple models.
    
    Args:
        histories (list): List of training history dictionaries
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.hsv(np.linspace(0, 0.6, len(histories)))
    
    # Determine the minimum number of epochs available for each model (train/val)
    for i, history in enumerate(histories):
        train_len = len(history['train_losses']) if history['train_losses'] is not None else 0
        val_len = len(history['val_losses']) if history['val_losses'] is not None else 0
        min_len = min(train_len, val_len) if (train_len > 0 and val_len > 0) else max(train_len, val_len)

        if history['train_losses'] is not None and min_len > 0:
            epochs = np.arange(1, min_len + 1)
            plt.semilogy(epochs, history['train_losses'][:min_len], 
                       color=colors[i], linestyle='-', alpha=0.7, linewidth=2,
                       label=f"{history['checkpoint_name']} (Train)")

        if history['val_losses'] is not None and min_len > 0:
            epochs = np.arange(1, min_len + 1)
            plt.semilogy(epochs, history['val_losses'][:min_len], 
                       color=colors[i], linestyle='--', alpha=0.9, linewidth=2,
                       label=f"{history['checkpoint_name']} (Val)")
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Comparison')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, which='both')
    plt.ylim(top=1e-2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def print_model_summary(histories):
    """
    Print a summary of model configurations and final performance.
    
    Args:
        histories (list): List of training history dictionaries
    """
    print("=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    for history in histories:
        config = history['config']
        checkpoint_name = history['checkpoint_name']
        
        print(f"\n{checkpoint_name}:")
        print("-" * 40)
        
        # Model info
        if 'model_class' in config:
            print(f"Model: {config['model_class']}")
        if 'dataset_class' in config:
            print(f"Dataset: {config['dataset_class']}")
        
        # Training info
        if 'num_epochs' in config:
            print(f"Epochs: {config['num_epochs']}")
        if 'batch_size' in config:
            print(f"Batch Size: {config['batch_size']}")
        if 'learning_rate' in config:
            print(f"Learning Rate: {config['learning_rate']}")
        if 'optimizer' in config:
            print(f"Optimizer: {config['optimizer']}")
        if 'criterion' in config:
            print(f"Loss Function: {config['criterion']}")
        
        # Performance
        if 'best_val_loss' in config:
            print(f"Best Val Loss: {config['best_val_loss']:.6f}")
        
        # Dataset info
        if 'train_size' in config and 'val_size' in config:
            print(f"Train/Val Split: {config['train_size']}/{config['val_size']}")
        
        # Final losses
        if history['train_losses'] is not None:
            final_train = history['train_losses'][-1]
            print(f"Final Train Loss: {final_train:.6f}")
        
        if history['val_losses'] is not None:
            final_val = history['val_losses'][-1]
            print(f"Final Val Loss: {final_val:.6f}")

def print_comparison_table(histories):
    """
    Print a formatted comparison table of all models.
    
    Args:
        histories (list): List of training history dictionaries
    """
    print("\n" + "=" * 120)
    print("COMPARISON TABLE")
    print("=" * 120)
    
    # Table header
    header = f"{'Model':<20} {'Dataset':<15} {'Best Val Loss':<12} {'Final Val Loss':<12} {'Epochs':<8} {'Batch':<6} {'LR':<8}"
    print(header)
    print("-" * 120)
    
    # Sort by best validation loss
    sorted_histories = sorted(histories, key=lambda h: h['config'].get('best_val_loss', float('inf')))
    
    for history in sorted_histories:
        config = history['config']
        checkpoint_name = history['checkpoint_name']
        
        model_class = config.get('model_class', 'Unknown')
        dataset_class = config.get('dataset_class', 'Unknown')
        best_val_loss = config.get('best_val_loss', float('inf'))
        final_val_loss = history['val_losses'][-1] if history['val_losses'] is not None else float('inf')
        epochs = config.get('num_epochs', 'Unknown')
        batch_size = config.get('batch_size', 'Unknown')
        lr = config.get('learning_rate', 'Unknown')
        
        row = f"{model_class:<20} {dataset_class:<15} {best_val_loss:<12.6f} {final_val_loss:<12.6f} {epochs:<8} {batch_size:<6} {lr:<8}"
        print(row)
    
    print("-" * 120)

def get_available_checkpoints():
    """
    Get list of available checkpoint directories.
    """
    checkpoint_dir = Path("/home/jamba/research/thesis-v2/mLearning/checkpoints")
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir():
            # Check if it has training history files
            if (item / 'train_losses.npy').exists() or (item / 'val_losses.npy').exists():
                checkpoints.append(item.name)
    
    return sorted(checkpoints)

def plot_sample_on_axes(y_true, y_pred, mask, metrics, idx, axes):
    """Plot predictions, errors, SDF, and time field for a single sample on provided axes."""
    
    # Set consistent vmin/vmax based on target 5th-95th percentile
    vmin = np.nanpercentile(y_true, 5)
    vmax = np.nanpercentile(y_true, 95)
    
    # Target
    im_target = axes[0].imshow(y_true, cmap='hot', vmin=vmin, vmax=vmax)
    # plt.colorbar(im_target, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Prediction
    im_pred = axes[1].imshow(y_pred, cmap='hot', vmin=vmin, vmax=vmax)
    # cbar_pred = plt.colorbar(im_pred, ax=axes[1], fraction=0.046, pad=0.04)
    # cbar_pred.set_label('Temperature (K)')

    # Percentage Error map
    perc_error_map = np.zeros_like(y_true)
    error_full = np.zeros_like(y_true)
    error_full[mask] = metrics['percent_error']
    perc_error_map = np.where(mask, error_full, np.nan)

    im0 = axes[2].imshow(perc_error_map, cmap='bwr', vmin=-2, vmax=2)
    # cbar_perc = plt.colorbar(im0, ax=axes[2], fraction=0.046, pad=0.04)
    # cbar_perc.set_label('Percent Error (%)')
    
    # Hide x and y axis labels and ticks for all axes, and remove plot boxes
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_visible(False)


def main():
    """
    Main function to compare multiple models.
    """
    # Get all available checkpoints
    available_checkpoints = get_available_checkpoints()
    print(f"Available checkpoints: {available_checkpoints}")
    
    # Define the checkpoint paths to compare
    # You can modify this list to include the models you want to compare
    # Or set to None to compare all available checkpoints
    selected_checkpoints = [
        "cnn_12_2025-08-04_23-09-22",
        "cnn_13_2025-08-04_22-56-59",
        "cnn_16_2025-08-05_02-20-31",
    ]
    # selected_checkpoints = None
    
    # If no specific checkpoints selected, use all available
    if selected_checkpoints is None:
        selected_checkpoints = available_checkpoints
    
    # Build full paths
    checkpoint_paths = [f"/home/jamba/research/thesis-v2/mLearning/checkpoints/{name}" for name in selected_checkpoints]
    
    # Load training histories
    histories = []
    for path in checkpoint_paths:
        if os.path.exists(path):
            history = load_training_history(path)
            if history['train_losses'] is not None or history['val_losses'] is not None:
                histories.append(history)
                print(f"Loaded: {path}")
        else:
            print(f"Warning: Checkpoint path not found: {path}")
    
    if not histories:
        print("No valid training histories found!")
        return
    
    # Print summary
    print_model_summary(histories)
    
    # Print comparison table
    print_comparison_table(histories)
    
    # Plot training curves
    plot_training_curves(histories, save_path="model_comparison.svg")
    
    # Additional analysis: convergence comparison
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)
    
    for history in histories:
        if history['val_losses'] is not None:
            val_losses = history['val_losses']
            final_loss = val_losses[-1]
            min_loss = np.min(val_losses)
            convergence_epoch = np.argmin(val_losses) + 1
            
            print(f"\n{history['checkpoint_name']}:")
            print(f"  Best Val Loss: {min_loss:.6f} (epoch {convergence_epoch})")
            print(f"  Final Val Loss: {final_loss:.6f}")
            print(f"  Improvement: {((final_loss - min_loss) / min_loss * 100):.2f}%")

if __name__ == "__main__":
    main()
