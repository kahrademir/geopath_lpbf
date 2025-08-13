"""
Database Evaluation Script for Thermal ML Models

This script evaluates trained models on individual data points within a specified database,
generating error visualizations and feature plots for each DATA_* directory.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import importlib
import math
import random
import re
from datetime import datetime
import sys

# Add the parent directory to the Python path
sys.path.append("/home/jamba/research/thesis-v2/")

# =============================================================================
# CONFIGURATION
# =============================================================================

# CONFIGURE THIS: Specify which database you want to evaluate
TARGET_DATABASE = 'DATABASE_PA_FILTERED_2'  # Change this to your desired database

# Data configuration
BASE_SIM_DATA_DIR = '/mnt/c/Users/jamba/sim_data/'
CHECKPOINT_BASE_DIR = 'mLearning/checkpoints'
GRID_SIZE = 192

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def find_available_checkpoints(base_dir=CHECKPOINT_BASE_DIR):
    """Find both single model checkpoints and k-fold checkpoint directories."""
    
    def extract_timestamp(path):
        """Extract timestamp from checkpoint directory name."""
        dir_name = os.path.basename(path)
        # Look for pattern like: cnn_7_2025-08-01_22-19-43 or 2025-07-28_00-34-22
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
        match = re.search(timestamp_pattern, dir_name)
        if match:
            try:
                timestamp_str = match.group(1)
                return datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
            except:
                pass
        return datetime.min  # Default for items without valid timestamp
    
    # Find single model checkpoints
    single_pattern = os.path.join(base_dir, '*/best_model.pth')
    single_checkpoints = glob.glob(single_pattern)
    single_checkpoints.sort(key=lambda x: extract_timestamp(os.path.dirname(x)), reverse=True)
    
    # Find k-fold checkpoint directories
    kfold_checkpoints = []
    for checkpoint_dir in glob.glob(os.path.join(base_dir, '*')):
        if os.path.isdir(checkpoint_dir):
            fold_dirs = glob.glob(os.path.join(checkpoint_dir, 'fold_*'))
            if fold_dirs:
                valid_folds = []
                for fold_dir in sorted(fold_dirs):
                    if os.path.exists(os.path.join(fold_dir, 'best_model.pth')):
                        valid_folds.append(fold_dir)
                if valid_folds:
                    kfold_checkpoints.append((checkpoint_dir, valid_folds))
    
    kfold_checkpoints.sort(key=lambda x: extract_timestamp(x[0]), reverse=True)
    return single_checkpoints, kfold_checkpoints

def select_checkpoint_and_mode(single_checkpoints, kfold_checkpoints):
    """Select checkpoint and evaluation mode (single model or k-fold)."""
    options = []
    
    # Add single model options
    for path in single_checkpoints:
        options.append(('single', path, None))
    
    # Add k-fold options
    for checkpoint_dir, fold_dirs in kfold_checkpoints:
        options.append(('kfold_all', checkpoint_dir, fold_dirs))
        for fold_dir in sorted(fold_dirs):
            fold_name = os.path.basename(fold_dir)
            options.append(('kfold_single', fold_dir, fold_name))
    
    if not options:
        raise FileNotFoundError("No valid checkpoints found in mLearning/checkpoints.")
    
    print("Available model checkpoints:")
    for idx, (mode, path, extra) in enumerate(options):
        if mode == 'single':
            print(f"  [{idx}] Single Model: {path}")
        elif mode == 'kfold_all':
            print(f"  [{idx}] K-Fold All Folds: {path} ({len(extra)} folds)")
        elif mode == 'kfold_single':
            print(f"  [{idx}] K-Fold {extra}: {path}")
    
    while True:
        try:
            selection = int(input(f"Select a checkpoint [0-{len(options)-1}]: "))
            if 0 <= selection < len(options):
                return options[selection]
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_components(eval_mode, checkpoint_path, extra_info):
    """Load config and model components based on evaluation mode."""
    
    # Determine config path based on evaluation mode
    if eval_mode == 'single':
        config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
    elif eval_mode in ['kfold_all', 'kfold_single']:
        if eval_mode == 'kfold_all':
            config_dir = checkpoint_path  # checkpoint_path is the parent dir
        else:
            config_dir = os.path.dirname(checkpoint_path)  # checkpoint_path is fold dir
        config_path = os.path.join(config_dir, 'config.json')
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get model module name
    model_module_name = config.get('model_module', 'cnn_1')
    if model_module_name == 'cnn_1':
        print("Warning: Using fallback model module cnn_1")
    
    # Import model components
    model_module = importlib.import_module(f'mLearning.models.{model_module_name}')
    
    return {
        'config': config,
        'model_module_name': model_module_name,
        'myModel': model_module.myModel,
        'myLoss': model_module.myLoss,
        'myDataset': model_module.myDataset,
        'myUnlabeledDataset': model_module.myUnlabeledDataset
    }

def load_single_model(model_path, model_class, device):
    """Load a single model from checkpoint."""
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_kfold_models(fold_dirs, model_class, device):
    """Load all models from k-fold directories."""
    models = {}
    for fold_dir in sorted(fold_dirs):
        fold_name = os.path.basename(fold_dir)
        model_path = os.path.join(fold_dir, 'best_model.pth')
        models[fold_name] = load_single_model(model_path, model_class, device)
        print(f"Loaded model for {fold_name}")
    return models

def prepare_models(eval_mode, checkpoint_path, extra_info, model_class, device):
    """Prepare models based on evaluation mode."""
    if eval_mode == 'single':
        model = load_single_model(checkpoint_path, model_class, device)
        models = {'single': model}
    elif eval_mode == 'kfold_single':
        model_path = os.path.join(checkpoint_path, 'best_model.pth')
        model = load_single_model(model_path, model_class, device)
        models = {extra_info: model}
    elif eval_mode == 'kfold_all':
        models = load_kfold_models(extra_info, model_class, device)
    else:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")
    
    return models

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_metrics(y_true, y_pred, mask):
    """Compute evaluation metrics for a single sample."""
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    abs_error = np.abs(y_true - y_pred)
    rel_error = np.divide(abs_error, np.abs(y_true), out=np.zeros_like(abs_error), where=np.abs(y_true)!=0) * 100
    mean_rel_error = np.mean(rel_error)
    max_rel_error = np.max(rel_error)
    
    return {
        'mse': mse, 'mae': mae, 'rmse': rmse,
        'mean_rel_error': mean_rel_error, 'max_rel_error': max_rel_error,
        'abs_error': abs_error, 'rel_error': rel_error
    }

def crop_to_mask_bounds(arrays, mask, padding=5):
    """Crop arrays to the bounding box of the mask with optional padding."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return arrays
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add padding
    rmin = max(0, rmin - padding)
    rmax = min(mask.shape[0], rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(mask.shape[1], cmax + padding + 1)
    
    # Crop all arrays
    cropped = []
    for arr in arrays:
        if arr is not None:
            cropped.append(arr[rmin:rmax, cmin:cmax])
        else:
            cropped.append(None)
    
    return cropped

def validate_database_path(database_name):
    """Validate and return the database path."""
    database_path = os.path.join(BASE_SIM_DATA_DIR, database_name)
    
    if not os.path.exists(database_path):
        available_databases = [
            os.path.basename(d) for d in glob.glob(os.path.join(BASE_SIM_DATA_DIR, 'DATABASE_*'))
        ]
        print(f"Error: Database '{database_name}' not found!")
        print(f"Available databases: {available_databases}")
        exit(1)
    
    return database_path

def save_evaluation_metadata(eval_dir_path, eval_mode, checkpoint_path, extra_info, components, database_name):
    """Save metadata about the evaluation for documentation."""
    from datetime import datetime
    
    metadata = {
        'evaluation_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'database': database_name,
            'evaluation_mode': eval_mode,
            'grid_size': GRID_SIZE
        },
        'model_info': {
            'model_module': components['model_module_name'],
            'checkpoint_path': os.path.abspath(checkpoint_path),
            'config': components['config']
        }
    }
    
    # Add specific info based on evaluation mode
    if eval_mode == 'single':
        metadata['model_info']['type'] = 'Single Model'
        metadata['model_info']['checkpoint_file'] = os.path.basename(checkpoint_path)
    elif eval_mode == 'kfold_single':
        metadata['model_info']['type'] = 'K-Fold Single'
        metadata['model_info']['fold_name'] = extra_info
        metadata['model_info']['fold_path'] = os.path.abspath(checkpoint_path)
    elif eval_mode == 'kfold_all':
        metadata['model_info']['type'] = 'K-Fold All Folds'
        metadata['model_info']['num_folds'] = len(extra_info)
        metadata['model_info']['fold_paths'] = [os.path.abspath(fold_dir) for fold_dir in extra_info]
    
    # Save as JSON file
    metadata_file = os.path.join(eval_dir_path, 'evaluation_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Also save a human-readable text summary
    summary_file = os.path.join(eval_dir_path, 'model_info.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("EVALUATION METADATA\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Evaluation Timestamp: {metadata['evaluation_info']['timestamp']}\n")
        f.write(f"Target Database: {database_name}\n")
        f.write(f"Evaluation Mode: {eval_mode}\n")
        f.write(f"Grid Size: {GRID_SIZE}\n\n")
        
        f.write("MODEL INFORMATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model Module: {components['model_module_name']}\n")
        f.write(f"Model Type: {metadata['model_info']['type']}\n")
        f.write(f"Main Checkpoint: {checkpoint_path}\n")
        
        if eval_mode == 'kfold_single':
            f.write(f"Fold Name: {extra_info}\n")
        elif eval_mode == 'kfold_all':
            f.write(f"Number of Folds: {len(extra_info)}\n")
            f.write("Fold Directories:\n")
            for i, fold_dir in enumerate(extra_info, 1):
                f.write(f"  {i}. {fold_dir}\n")
        
        f.write(f"\nCONFIGURATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Target Database: {TARGET_DATABASE}\n")
        f.write(f"Base Data Directory: {BASE_SIM_DATA_DIR}\n")
        f.write(f"Checkpoint Base Directory: {CHECKPOINT_BASE_DIR}\n")
        
        # Add key config parameters if available
        if 'dataset_kwargs' in components['config']:
            f.write(f"\nDataset Configuration:\n")
            for key, value in components['config']['dataset_kwargs'].items():
                if key == 'data_dirs' and isinstance(value, list) and len(value) > 3:
                    f.write(f"  {key}: [{len(value)} directories]\n")
                else:
                    f.write(f"  {key}: {value}\n")
    
    print(f"Saved evaluation metadata to:")
    print(f"  - {metadata_file}")
    print(f"  - {summary_file}")
    
    return metadata_file, summary_file

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_feature_visualization(dataset, save_dir, data_name):
    """Plot the input features (SDF, time field, gradient magnitude)."""
    if len(dataset) == 0:
        print("Dataset is empty, skipping feature visualization")
        return
    
    # Get a sample from the dataset
    sample = dataset[0]
    input_tensor = sample['input'].cpu().numpy()  # Shape: (C, H, W)
    mask = sample['mask'].cpu().numpy().astype(bool)
    
    # Determine feature names based on number of channels
    n_channels = input_tensor.shape[0]
    if n_channels == 2:
        feature_names = ['SDF', 'Time Field']
    elif n_channels == 3:
        feature_names = ['SDF', 'Time Field', 'Gradient Magnitude']
    else:
        feature_names = [f'Channel {i+1}' for i in range(n_channels)]
    
    # Create subplots
    fig, axes = plt.subplots(1, n_channels, figsize=(4*n_channels, 4))
    if n_channels == 1:
        axes = [axes]
    
    for i in range(n_channels):
        feature_data = input_tensor[i]
        feature_masked = np.where(mask, feature_data, np.nan)
        feature_crop = crop_to_mask_bounds([feature_masked], mask, padding=10)[0]
        
        im = axes[i].imshow(feature_crop, cmap='viridis')
        axes[i].set_title(f'{feature_names[i]}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    
    # Save the visualization
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        feature_filename = f'features_{data_name}.png'
        plt.savefig(os.path.join(save_dir, feature_filename), dpi=150, bbox_inches='tight')
        print(f"Saved feature visualization to {os.path.join(save_dir, feature_filename)}")
    
    plt.close()

def plot_sample_on_axes(y_true, y_pred, mask, metrics, idx, axes):
    """Plot predictions, errors, and metrics for a single sample."""
    # Crop arrays to focus on the data region
    y_true_crop, y_pred_crop, mask_crop = crop_to_mask_bounds([y_true, y_pred, mask], mask, padding=10)
    
    # Determine common vmin/vmax for target and prediction
    vmin = np.nanpercentile(y_true_crop, 5)
    vmax = np.nanpercentile(y_true_crop, 95)
    
    # Target
    im_target = axes[0].imshow(y_true_crop, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Sample {idx+1} Target (K)')
    
    # Prediction
    im_pred = axes[1].imshow(y_pred_crop, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Sample {idx+1} Prediction (K)')
    plt.colorbar(im_target, ax=axes[1])
    
    # Percentage Error map
    perc_error_map = np.zeros_like(y_true_crop)
    mask_flat = mask.flatten()
    if np.any(mask_flat):
        error_full = np.zeros_like(y_true)
        error_full[mask] = metrics['rel_error']
        error_crop = crop_to_mask_bounds([error_full], mask, padding=10)[0]
        perc_error_map = np.where(mask_crop, error_crop, np.nan)
    else:
        perc_error_map = np.where(mask_crop, perc_error_map, np.nan)
    
    perc_vmax = np.percentile(metrics['rel_error'], 99) if len(metrics['rel_error']) > 0 else 1
    im0 = axes[2].imshow(perc_error_map, cmap='Reds', vmin=0, vmax=perc_vmax)
    axes[2].set_title(f'Sample {idx+1} Percentage Error (%)')
    plt.colorbar(im0, ax=axes[2])
    
    # Clean up axes
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_single_model(model, dataset, device, num_samples=2, model_name="Model"):
    """Evaluate a single model and return metrics and sample data."""
    all_metrics = []
    sample_data = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample_idx = random.randint(0, len(dataset) - 1)
            sample = dataset[sample_idx]
            inputs = sample['input'].unsqueeze(0).to(device)
            target = sample['target'].cpu().numpy()
            mask = sample['mask'].cpu().numpy().astype(bool)
            prediction = model(inputs).squeeze().cpu().numpy()
            
            # Denormalize for visualization
            target_denorm = dataset.denormalize_fields([target], minmax=[1200, 1800])[0]
            pred_denorm = dataset.denormalize_fields([prediction], minmax=[1200, 1800])[0]
            target_denorm = np.where(mask, target_denorm, np.nan)
            pred_denorm = np.where(mask, pred_denorm, np.nan)
            
            metrics = compute_metrics(target_denorm, pred_denorm, mask)
            all_metrics.append(metrics)
            sample_data.append((target_denorm, pred_denorm, mask, metrics))
    
    return all_metrics, sample_data

def evaluate_and_visualize_multi(models, dataset, device, eval_mode, num_samples=6, save_dir=None, data_name=None):
    """Evaluate multiple models and create visualizations."""
    all_results = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\n=== EVALUATING {model_name.upper()} ===")
        all_metrics, sample_data = evaluate_single_model(model, dataset, device, num_samples, model_name)
        
        # Print per-sample results
        for i, metrics in enumerate(all_metrics):
            print(f"Sample {i+1}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
                  f"RMSE={metrics['rmse']:.4f}, Mean Rel Err={metrics['mean_rel_error']:.2f}%, "
                  f"Max Rel Err={metrics['max_rel_error']:.2f}%")
        
        # Aggregate metrics
        mse = np.mean([m['mse'] for m in all_metrics])
        mae = np.mean([m['mae'] for m in all_metrics])
        rmse = np.mean([m['rmse'] for m in all_metrics])
        mean_rel_error = np.mean([m['mean_rel_error'] for m in all_metrics])
        max_rel_error = np.max([m['max_rel_error'] for m in all_metrics])
        
        all_results[model_name] = {
            'metrics': all_metrics,
            'sample_data': sample_data,
            'aggregated': {
                'mse': mse, 'mae': mae, 'rmse': rmse,
                'mean_rel_error': mean_rel_error, 'max_rel_error': max_rel_error
            }
        }
        
        print(f"\n=== SUMMARY FOR {model_name.upper()} ===")
        print(f"Mean MSE: {mse:.4f}")
        print(f"Mean MAE: {mae:.4f}")
        print(f"Mean RMSE: {rmse:.4f}")
        print(f"Mean Relative Error: {mean_rel_error:.2f}%")
        print(f"Max Relative Error: {max_rel_error:.2f}%")
    
    # Print cross-fold summary if multiple models
    if len(models) > 1:
        print(f"\n=== CROSS-FOLD SUMMARY ===")
        metrics_names = ['mse', 'mae', 'rmse', 'mean_rel_error', 'max_rel_error']
        for metric in metrics_names:
            values = [all_results[name]['aggregated'][metric] for name in all_results.keys()]
            mean_val = np.mean(values)
            std_val = np.std(values)
            if 'error' in metric:
                print(f"{metric.upper()}: {mean_val:.2f}% ± {std_val:.2f}%")
            else:
                print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Create visualization
    first_model = list(models.keys())[0]
    sample_data = all_results[first_model]['sample_data']
    
    n_samples = len(sample_data)
    ncols = 3
    nrows = n_samples
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.2*ncols, 2.2*nrows))
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.rcParams.update({'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7})
    
    # Handle single row case
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot per-sample plots
    for i, (target, pred, mask, metrics) in enumerate(sample_data):
        plot_sample_on_axes(target, pred, mask, metrics, i, axes[i, :])
    
    plt.tight_layout(pad=1.0)
    
    # Save visualization
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if data_name:
            filename = f'eval_{data_name}_{eval_mode}.png'
        else:
            filename = f'eval_{eval_mode}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {os.path.join(save_dir, filename)}")
    
    plt.close()
    return all_results

def evaluate_data_directory(data_dir, eval_dir_path, models, device, eval_mode, dataset_class):
    """Evaluate a single data directory."""
    data_name = os.path.basename(data_dir)
    print(f"\n--- Evaluating {data_name} ---")
    
    # Create dataset for this single data directory
    dataset_kwargs = {
        'data_dirs': [data_dir],
        'N_max': GRID_SIZE
    }
    
    try:
        dataset = dataset_class(**dataset_kwargs)
        print(f"Successfully created dataset for {data_name} with {len(dataset)} samples")
    except Exception as e:
        print(f"Error creating dataset for {data_name}: {e}")
        return False
    
    # Run evaluation
    evaluation_results = evaluate_and_visualize_multi(
        models, dataset, device, eval_mode,
        num_samples=min(6, len(dataset)),
        save_dir=eval_dir_path,
        data_name=data_name
    )
    
    # Plot feature visualization
    plot_feature_visualization(dataset, eval_dir_path, data_name)
    
    print(f"Completed evaluation for {data_name}")
    return True

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=== THERMAL ML MODEL DATABASE EVALUATION ===\n")
    
    # 1. Checkpoint Selection
    print("1. Selecting model checkpoint...")
    single_checkpoints, kfold_checkpoints = find_available_checkpoints()
    eval_mode, checkpoint_path, extra_info = select_checkpoint_and_mode(single_checkpoints, kfold_checkpoints)
    print(f"Selected: {eval_mode} - {checkpoint_path}")
    if extra_info and eval_mode == 'kfold_all':
        print(f"Will evaluate {len(extra_info)} folds")
    
    # 2. Load Model Components
    print(f"\n2. Loading model components...")
    components = load_model_components(eval_mode, checkpoint_path, extra_info)
    print(f"Using model module: {components['model_module_name']}")
    
    # 3. Database Validation
    print(f"\n3. Validating target database...")
    database_path = validate_database_path(TARGET_DATABASE)
    print(f"Target database: {TARGET_DATABASE}")
    
    # 4. Setup Device and Models
    print(f"\n4. Setting up device and models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 5. Database Evaluation Setup
    print(f"\n5. Setting up evaluation environment...")
    database_name = os.path.basename(database_path)
    eval_dir_name = f"eval_{database_name}"
    eval_dir_path = os.path.join(os.path.dirname(database_path), eval_dir_name)
    os.makedirs(eval_dir_path, exist_ok=True)
    print(f"Created evaluation directory: {eval_dir_path}")
    
    # Save evaluation metadata
    save_evaluation_metadata(eval_dir_path, eval_mode, checkpoint_path, extra_info, components, database_name)
    
    # Find all DATA_* directories
    data_dirs = glob.glob(os.path.join(database_path, 'DATA_*'))
    data_dirs.sort()
    
    if not data_dirs:
        print(f"No DATA_* directories found in {database_name}")
        return False
    
    print(f"Found {len(data_dirs)} data directories in {database_name}")
    
    # 6. Evaluate Each Data Directory
    print(f"\n6. Starting evaluation of data directories...")
    print(f"{'='*50}")
    print(f"EVALUATING DATABASE: {database_name}")
    print(f"{'='*50}")
    
    successful_evaluations = 0
    
    models = prepare_models(eval_mode, checkpoint_path, extra_info, components['myModel'], device)
    for data_dir in data_dirs:
        # Evaluate this data directory
        success = evaluate_data_directory(
            data_dir, eval_dir_path, models, device, eval_mode, components['myDataset']
        )
        
        if success:
            successful_evaluations += 1
    
    # 7. Summary
    print(f"\n{'='*50}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*50}")
    print(f"Database: {database_name}")
    print(f"Successful evaluations: {successful_evaluations}/{len(data_dirs)}")
    print(f"Results saved to: {eval_dir_path}")
    
    return True

if __name__ == "__main__":
    main()