# %%
"""
Model Evaluation Script for Thermal Prediction Models

This script provides functionality to:
1. Load a trained model from a checkpoint directory
2. Evaluate the model's error metrics on a given dataset  
3. Plot prediction results and error distributions

Usage:
1. Update CHECKPOINT_PATH below to point to your model checkpoint
2. Run the script to get evaluation results and visualizations
3. Optionally uncomment sections for detailed error analysis or unlabeled predictions

Key Functions:
- evaluate_model(): Core evaluation function that computes metrics
- plot_evaluation_results(): Creates visualization plots
- plot_error_histograms(): Detailed error distribution analysis  
- predict_and_plot_grid(): Make predictions on unlabeled data
"""
import json
from shapely import wkt
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import importlib
import math
import random
import sys
import matplotlib as mpl

mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
# mpl.rcParams['colorbar.labelsize'] = 16
# mpl.rcParams['colorbar.tick.labelsize'] = 16

sys.path.append("/home/jamba/research/thesis-v2")


def load_model_and_dataset(checkpoint_path, data_dirs=None):
    """Load model, configuration, and dataset from checkpoint directory."""
    # If checkpoint_path is a .pth file, get its directory
    if checkpoint_path.endswith('.pth'):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        model_path = checkpoint_path
    else:
        checkpoint_dir = checkpoint_path
        model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # Load configuration
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load the model module from config
    MODEL_MODULE = config['model_module']
    
    # Dynamic import based on MODEL_MODULE from config
    model_module = importlib.import_module(f'mLearning.models.{MODEL_MODULE}')
    myModel = model_module.myModel
    model = myModel().to('cuda')
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()
    
    myLoss = model_module.myLoss
    myDataset = model_module.myDataset
    # myUnlabeledDataset = model_module.myUnlabeledDataset

    # Create dataset using configuration
    dataset_kwargs = config['dataset_kwargs']
    if data_dirs is not None:
        dataset_kwargs['data_dirs'] = data_dirs
        dataset_kwargs['N_max'] = None
    dataset = myDataset(**dataset_kwargs)
    
    return model, dataset, myLoss, model_path, config

def compute_error(y_true, y_pred, mask):
    """Compute evaluation metrics for a single sample."""
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    error = y_pred - y_true
    percent_error = np.divide(error, y_true, out=np.zeros_like(error), where=y_true!=0) * 100
    mse = np.mean(error ** 2)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(mse)
    abs_error = np.abs(error)
    # Relative error (%) measures the absolute error as a percentage of the true value for each pixel
    rel_error = np.divide(abs_error, np.abs(y_true), out=np.zeros_like(abs_error), where=np.abs(y_true)!=0) * 100
    mean_rel_error = np.mean(rel_error)
    max_rel_error = np.max(rel_error)
    return {
        'error': error,
        'percent_error': percent_error,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mean_rel_error': mean_rel_error,
        'max_rel_error': max_rel_error,
        'abs_error': abs_error,
        'rel_error': rel_error
    }

def plot_sample_on_axes(y_true, y_pred, mask, metrics, idx, axes):
    """Plot predictions, errors, SDF, and time field for a single sample on provided axes."""
    
    # Set consistent vmin/vmax based on target 5th-95th percentile
    vmin = np.nanpercentile(y_true, 1)
    # vmax = np.nanpercentile(y_true, 99)
    vmax = 1800
    
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

    im0 = axes[2].imshow(perc_error_map, cmap='bwr', vmin=-4, vmax=4)
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

def evaluate_model(model, dataset, num_samples=6, save_dir=None, indices=None):
    """Evaluate a model and return metrics and sample data.

    Args:
        model: The trained model to evaluate.
        dataset: The dataset to sample from.
        num_samples: Number of samples to evaluate (ignored if indices is provided).
        save_dir: Directory to save results (optional).
        indices: Optional list of sample indices to use instead of random sampling.
    """
    all_metrics = []
    sample_data = []

    if indices is not None:
        sample_indices = indices
    else:
        sample_indices = [random.randint(0, len(dataset) - 1) for _ in range(num_samples)]

    with torch.no_grad():
        for sample_idx in sample_indices:
            sample = dataset[sample_idx]
            inputs = sample['input'].unsqueeze(0).to('cuda')
            target = sample['target'].cpu().numpy()
            mask = sample['mask'].cpu().numpy().astype(bool)
            prediction = model(inputs).squeeze().cpu().numpy()

            # Denormalize for visualization
            target_denorm = dataset.denormalize_fields([target], minmax=[1000, 2000])[0]
            pred_denorm = dataset.denormalize_fields([prediction], minmax=[1000, 2000])[0]

            # Set values outside the mask to NaN for visualization
            target_denorm = np.where(mask, target_denorm, np.nan)
            pred_denorm = np.where(mask, pred_denorm, np.nan)

            metrics = compute_error(target_denorm, pred_denorm, mask)
            all_metrics.append(metrics)
            sample_data.append((target_denorm, pred_denorm, mask, metrics))

    return all_metrics, sample_data

def plot_evaluation_results(sample_data, save_dir=None):
    """Create visualization plots for evaluation results."""
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
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = 'evaluation_results.svg'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight', format='svg')
    plt.show()

def plot_error_histograms(all_metrics, save_dir=None):
    """Plot histograms of different error metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Collect all error values
    all_abs_errors = []
    all_rel_errors = []
    sample_mses = [m['mse'] for m in all_metrics]
    sample_maes = [m['mae'] for m in all_metrics]
    sample_rmses = [m['rmse'] for m in all_metrics]
    sample_mean_rel_errors = [m['mean_rel_error'] for m in all_metrics]
    
    # Collect all individual pixel errors
    for m in all_metrics:
        all_abs_errors.extend(m['abs_error'])
        all_rel_errors.extend(m['rel_error'])
    
    # Plot histograms
    # Absolute errors (all pixels)
    axes[0, 0].hist(all_abs_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Absolute Errors (K)')
    axes[0, 0].set_xlabel('Absolute Error (K)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Relative errors (all pixels) - limit to reasonable range
    rel_errors_capped = [e for e in all_rel_errors if e <= 100]  # Cap at 100%
    axes[0, 1].hist(rel_errors_capped, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_title('Distribution of Relative Errors (%)')
    axes[0, 1].set_xlabel('Relative Error (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # MSE per sample
    axes[0, 2].hist(sample_mses, bins=20, alpha=0.7, edgecolor='black', color='red')
    axes[0, 2].set_title('Distribution of Sample MSE')
    axes[0, 2].set_xlabel('MSE')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # MAE per sample
    axes[1, 0].hist(sample_maes, bins=20, alpha=0.7, edgecolor='black', color='green')
    axes[1, 0].set_title('Distribution of Sample MAE')
    axes[1, 0].set_xlabel('MAE (K)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # RMSE per sample
    axes[1, 1].hist(sample_rmses, bins=20, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 1].set_title('Distribution of Sample RMSE')
    axes[1, 1].set_xlabel('RMSE (K)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Mean relative error per sample
    axes[1, 2].hist(sample_mean_rel_errors, bins=20, alpha=0.7, edgecolor='black', color='brown')
    axes[1, 2].set_title('Distribution of Sample Mean Relative Error')
    axes[1, 2].set_xlabel('Mean Relative Error (%)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'error_histograms.png'), dpi=150, bbox_inches='tight')
    
    plt.show()

def run_detailed_error_analysis(model, dataset, num_samples=50, save_dir=None):
    """Run detailed error analysis with histograms."""
    detailed_metrics = []
    
    with torch.no_grad():
        for i in range(num_samples):
            sample_idx = random.randint(0, len(dataset) - 1)
            sample = dataset[sample_idx]
            inputs = sample['input'].unsqueeze(0).to('cuda')
            target = sample['target'].cpu().numpy()
            mask = sample['mask'].cpu().numpy().astype(bool)
            prediction = model(inputs).squeeze().cpu().numpy()
            
            # Denormalize for evaluation
            target_denorm = dataset.denormalize_fields([target], minmax=[1000, 2000])[0]
            pred_denorm = dataset.denormalize_fields([prediction], minmax=[1000, 2000])[0]
            
            # Set values outside the mask to NaN for evaluation
            target_denorm = np.where(mask, target_denorm, np.nan)
            pred_denorm = np.where(mask, pred_denorm, np.nan)
            
            metrics = compute_error(target_denorm, pred_denorm, mask)
            detailed_metrics.append(metrics)
    
    plot_error_histograms(detailed_metrics, save_dir=save_dir)
    return detailed_metrics

def error_vs_pa(model, dataset, num_samples=None, save_dir=None):
    """
    Plot the error vs the perimeter-to-area ratio of the target shape.
    For each sample, finds the corresponding metadata_00.json file in the sample's data_dir,
    reads the WKT Polygon, and computes the perimeter-to-area ratio.
    """


    if num_samples is None:
        num_samples = len(dataset)
    all_errors = []
    all_pas = []

    # Helper to get P/A ratio from metadata_00.json in a data_dir
    def get_pa_ratio_from_metadata(data_dir):
        meta_path = os.path.join(data_dir, "metadata_00.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        poly = wkt.loads(meta["shape"])
        area = poly.area
        perimeter = poly.length
        return perimeter / area

    with torch.no_grad():
        for i in range(num_samples):
            # Get sample index
            if num_samples is None:
                sample_idx = i
            else:
                sample_idx = random.randint(0, len(dataset) - 1)
            sample = dataset[sample_idx]
            inputs = sample['input'].unsqueeze(0).to('cuda')
            target = sample['target'].cpu().numpy()
            mask = sample['mask'].cpu().numpy().astype(bool)

            # Find the data_dir for this sample
            # Assumes dataset has a list/attribute mapping indices to data_dirs
            if hasattr(dataset, "data_dirs"):
                data_dir = dataset.data_dirs[sample_idx]
            elif hasattr(dataset, "samples") and "data_dir" in dataset.samples[sample_idx]:
                data_dir = dataset.samples[sample_idx]["data_dir"]
            else:
                raise RuntimeError("Cannot determine data_dir for sample.")

            pa_ratio = get_pa_ratio_from_metadata(data_dir)
            all_pas.append(pa_ratio)

            # Compute error (example: RMSE over mask)
            prediction = model(inputs).squeeze().cpu().numpy()
            target_masked = target[mask]
            pred_masked = prediction[mask]
            rmse = np.sqrt(np.mean((target_masked - pred_masked) ** 2))
            all_errors.append(rmse)

    print(f"Found {len(all_errors)} samples")
    return all_errors, all_pas

checkpoints = [
    "cnn_12_2025-08-04_23-09-22",
    "cnn_13_2025-08-04_22-56-59",
    "cnn_16_2025-08-05_02-20-31",
]
CHECKPOINT_PATH = f"/home/jamba/research/thesis-v2/mLearning/checkpoints/{checkpoints[2]}/best_model.pth"  # Update this path

# Load model and configuration
data_dirs = glob.glob(os.path.join('/mnt/c/Users/jamba/sim_data/DATABASE_PA/', 'DATA_*'))
# data_dirs.extend(glob.glob(os.path.join('/mnt/c/Users/jamba/sim_data/DATABASE_UNIRASTER/', 'DATA_*')))
# data_dirs.extend(glob.glob(os.path.join('/mnt/c/Users/jamba/sim_data/DATABASE_FULL/', 'DATA_*')))

model, dataset, myLoss, model_path, config = load_model_and_dataset(CHECKPOINT_PATH, data_dirs=data_dirs)
# %%

if __name__ == "__main__":
    # indices = [69, 70, 106]
    indices = [82, 87, 127]

    all_metrics, sample_data = evaluate_model(model, dataset, save_dir='eval_results', indices=indices)
    plot_evaluation_results(sample_data, save_dir='eval_results')
    # all_errors, all_pas = error_vs_pa(model, dataset, save_dir='eval_results')
        # detailed_metrics = run_detailed_error_analysis(model, dataset, num_samples=50, save_dir='eval_results')
    # %%
    # plt.figure(figsize=(7,8))
    # plt.scatter(all_pas, all_errors, color='black', s=20)
    # plt.xlabel('Perimeter-to-Area Ratio')
    # plt.ylabel('RMSE')
    # # plt.title('RMSE vs Perimeter-to-Area Ratio')
    # plt.tight_layout()
    # plt.show()

# %%
