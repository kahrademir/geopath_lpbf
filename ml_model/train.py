"""
K-Fold Cross Validation Training Script

This script supports both traditional single train/val split training and k-fold cross validation.

Usage Examples:
    # Standard single split training (default)
    python train.py --num_epochs 50

    # K-fold cross validation with 5 folds
    python train.py --k_folds 5 --num_epochs 30
    
    # Resume specific fold
    python train.py --resume_dir checkpoints/cnn_7_2024-01-01_12-00-00 --resume_fold 2
    
    # Resume all incomplete folds
    python train.py --resume_dir checkpoints/cnn_7_2024-01-01_12-00-00

Features:
    - Maintains backward compatibility with original single-split training
    - Reproducible k-fold splits using sklearn KFold with fixed random state
    - Individual model checkpoints for each fold (fold_0/, fold_1/, etc.)
    - Comprehensive results aggregation and visualization
    - Resume support for individual folds or entire cross-validation runs
    - Statistical summaries (mean, std) across all folds
"""

import torch
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
import os
import json
from datetime import datetime
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import importlib
from sklearn.model_selection import KFold

# === CONFIGURATION ===
# Change this line to switch models (e.g., 'cnn_1', 'cnn_2', etc.)
MODEL_MODULE = 'cnn_16'

# Dynamic import based on MODEL_MODULE
model_module = importlib.import_module(f'mLearning.models.{MODEL_MODULE}')
myModel = model_module.myModel
myLoss = model_module.myLoss
myDataset = model_module.myDataset
myUnlabeledDataset = model_module.myUnlabeledDataset

# Change these lines to switch datasets
# data_dirs = glob.glob(os.path.join('/mnt/c/Users/jamba/sim_data/DATABASE_001/', 'DATA_*'))
data_dirs = glob.glob(os.path.join('/mnt/c/Users/jamba/sim_data/DATABASE_PA/', 'DATA_*'))
# data_dirs.extend(glob.glob(os.path.join('/mnt/c/Users/jamba/sim_data/DATABASE_PA/', 'DATA_*')))
data_dirs.extend(glob.glob(os.path.join('/mnt/c/Users/jamba/sim_data/DATABASE_PA_FILTERED_2/', 'DATA_*')))

print(f"Found {len(data_dirs)} data directories")
dataset_kwargs = {
    'data_dirs': data_dirs,
    'N_max': 192
}

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description='Train or resume a model.')
parser.add_argument('--resume_dir', type=str, default=None, help='Path to checkpoint directory to resume from')
parser.add_argument('--num_epochs', type=int, default=None, help='Total number of epochs to train (can be used to extend training when resuming)')
parser.add_argument('--k_folds', type=int, default=None, help='Number of folds for k-fold cross validation (default: None for single split)')
parser.add_argument('--resume_fold', type=int, default=None, help='Specific fold to resume (use with --resume_dir)')
args = parser.parse_args()

# === CHECKPOINT DIRECTORY & CONFIG SETUP ===
if args.resume_dir:
    checkpoint_dir = args.resume_dir
    with open(os.path.join(checkpoint_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    start_epoch = config.get('last_epoch', 0) + 1
    best_val_loss = config.get('best_val_loss', float('inf'))
    
    # Load the model module from config if available, otherwise fall back to current MODEL_MODULE
    if 'model_module' in config:
        MODEL_MODULE = config['model_module']
        # Re-import the correct model module
        model_module = importlib.import_module(f'mLearning.models.{MODEL_MODULE}')
        myModel = model_module.myModel
        myLoss = model_module.myLoss
        myDataset = model_module.myDataset
        myUnlabeledDataset = model_module.myUnlabeledDataset
    
    dataset_kwargs = config['dataset_kwargs']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    device = torch.device(config['device'])
    learning_rate = config['learning_rate']
    # Load existing training history
    train_losses_file = os.path.join(checkpoint_dir, 'train_losses.npy')
    val_losses_file = os.path.join(checkpoint_dir, 'val_losses.npy')
    if os.path.exists(train_losses_file) and os.path.exists(val_losses_file):
        existing_train_losses = np.load(train_losses_file).tolist()
        existing_val_losses = np.load(val_losses_file).tolist()
        print(f"Loaded existing training history: {len(existing_train_losses)} epochs")
    else:
        existing_train_losses = []
        existing_val_losses = []
        print("No existing training history found, starting fresh")
    
    # If user provides a new num_epochs, update config and use it
    if args.num_epochs is not None and args.num_epochs > num_epochs:
        num_epochs = args.num_epochs
        config['num_epochs'] = num_epochs
        with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
else:
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_base = os.path.join('mLearning', 'checkpoints')
    checkpoint_dir = os.path.join(checkpoint_base, f'{MODEL_MODULE}_{run_timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = 1
    best_val_loss = float('inf')
    batch_size = 16
    num_epochs = args.num_epochs if args.num_epochs is not None else 50
    device = torch.device('cuda')
    learning_rate = 2.5e-4
    # Initialize empty training history for new runs
    existing_train_losses = []
    existing_val_losses = []
    # Read the entire model source file
    model_source_path = f'mLearning/models/{MODEL_MODULE}.py'
    with open(model_source_path, 'r') as f:
        model_source_code = f.read()
    
    config = {
        'model_class_name': myModel.__name__,
        'model_module': MODEL_MODULE,
        'model_source_file': model_source_path,
        'model_source_code': model_source_code,
        'dataset_kwargs': dataset_kwargs,
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'device': str(device),
        'last_epoch': 0,
        'best_val_loss': best_val_loss
    }

# === DATASET & DATALOADERS ===
dataset = myDataset(**dataset_kwargs)
if args.resume_dir:
    train_size = config['train_size']
    val_size = config['val_size']
else:
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    config['train_size'] = train_size
    config['val_size'] = val_size
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === MODEL, CRITERION, OPTIMIZER ===
model = myModel().to(device)
criterion = myLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# Add save interval for best model
save_interval = 5  # Save best model at most every 5 epochs
last_best_save_epoch = start_epoch - 1

if args.resume_dir:
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), map_location=device))
    optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pth')
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

# === HELPER FUNCTIONS ===
def create_k_fold_datasets(dataset, k_folds, random_seed=42):
    """Create k-fold datasets for cross validation."""
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    fold_datasets = []
    
    indices = list(range(len(dataset)))
    for train_indices, val_indices in kfold.split(indices):
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        fold_datasets.append((train_dataset, val_dataset))
    
    return fold_datasets

def train_single_fold(fold_num, train_dataset, val_dataset, config, checkpoint_dir, device):
    """Train a single fold and return the results."""
    print(f"\n=== TRAINING FOLD {fold_num + 1}/{config.get('k_folds', 1)} ===")
    
    # Create fold-specific checkpoint directory
    fold_checkpoint_dir = os.path.join(checkpoint_dir, f'fold_{fold_num}')
    os.makedirs(fold_checkpoint_dir, exist_ok=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model, criterion, optimizer
    model = myModel().to(device)
    criterion = myLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Resume fold if checkpoint exists
    start_epoch = 1
    best_val_loss = float('inf')
    existing_train_losses = []
    existing_val_losses = []
    
    fold_config_path = os.path.join(fold_checkpoint_dir, 'config.json')
    if os.path.exists(fold_config_path):
        with open(fold_config_path, 'r') as f:
            fold_config = json.load(f)
        start_epoch = fold_config.get('last_epoch', 0) + 1
        best_val_loss = fold_config.get('best_val_loss', float('inf'))
        
        # Load model and optimizer if they exist
        model_path = os.path.join(fold_checkpoint_dir, 'best_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        optimizer_path = os.path.join(fold_checkpoint_dir, 'optimizer.pth')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        
        # Load existing training history
        train_losses_file = os.path.join(fold_checkpoint_dir, 'train_losses.npy')
        val_losses_file = os.path.join(fold_checkpoint_dir, 'val_losses.npy')
        if os.path.exists(train_losses_file) and os.path.exists(val_losses_file):
            existing_train_losses = np.load(train_losses_file).tolist()
            existing_val_losses = np.load(val_losses_file).tolist()
    
    # Training loop for this fold
    train_losses = existing_train_losses.copy()
    val_losses = existing_val_losses.copy()
    save_interval = 5
    last_best_save_epoch = start_epoch - 1
    
    for epoch in range(start_epoch, config['num_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        print(f"Fold {fold_num + 1}, Epoch {epoch}: Train Loss = {train_loss:.2e}, Val Loss = {val_loss:.2e}, LR = {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if (epoch - last_best_save_epoch) >= save_interval or epoch == config['num_epochs']:
                torch.save(model.state_dict(), os.path.join(fold_checkpoint_dir, 'best_model.pth'))
                last_best_save_epoch = epoch
                print(f"Saved new best model for fold {fold_num + 1} at epoch {epoch}.")
        
        # Save optimizer and update config
        torch.save(optimizer.state_dict(), os.path.join(fold_checkpoint_dir, 'optimizer.pth'))
        
        fold_config = {
            'fold_num': fold_num,
            'last_epoch': epoch,
            'best_val_loss': best_val_loss,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset)
        }
        with open(fold_config_path, 'w') as f:
            json.dump(fold_config, f, indent=2)
    
    # Save training history
    np.save(os.path.join(fold_checkpoint_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(fold_checkpoint_dir, 'val_losses.npy'), np.array(val_losses))
    
    # Plot fold-specific loss curve
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss', alpha=0.8)
    plt.plot(epochs, val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold_num + 1} - Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(fold_checkpoint_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()  # Close to avoid memory issues
    
    return {
        'fold_num': fold_num,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'train_losses': train_losses,
        'val_losses': val_losses
    }

# === TRAINING LOOP ===
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        masks = batch['mask'].to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            masks = batch['mask'].to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# === MAIN TRAINING LOGIC ===
# Check if k-fold cross validation is requested
if args.k_folds is not None:
    print(f"\n=== K-FOLD CROSS VALIDATION (k={args.k_folds}) ===")
    
    # Update config for k-fold mode
    if not args.resume_dir:
        config['k_folds'] = args.k_folds
        config['cv_mode'] = True
        with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    # Create k-fold datasets
    fold_datasets = create_k_fold_datasets(dataset, args.k_folds)
    
    # Track results across all folds
    fold_results = []
    
    # Determine which folds to train
    if args.resume_fold is not None:
        folds_to_train = [args.resume_fold]
        print(f"Resuming training for fold {args.resume_fold}")
    else:
        folds_to_train = range(args.k_folds)
    
    # Train each fold
    for fold_num in folds_to_train:
        train_dataset_fold, val_dataset_fold = fold_datasets[fold_num]
        print(f"\nFold {fold_num + 1}: Train size = {len(train_dataset_fold)}, Val size = {len(val_dataset_fold)}")
        
        fold_result = train_single_fold(fold_num, train_dataset_fold, val_dataset_fold, 
                                      config, checkpoint_dir, device)
        fold_results.append(fold_result)
    
    # Aggregate and display results
    if len(fold_results) == args.k_folds:  # Only if all folds are complete
        print(f"\n=== K-FOLD CROSS VALIDATION RESULTS ===")
        
        val_losses_all_folds = [result['best_val_loss'] for result in fold_results]
        train_losses_all_folds = [result['final_train_loss'] for result in fold_results]
        
        print(f"Validation Loss - Mean: {np.mean(val_losses_all_folds):.6f}, Std: {np.std(val_losses_all_folds):.6f}")
        print(f"Training Loss - Mean: {np.mean(train_losses_all_folds):.6f}, Std: {np.std(train_losses_all_folds):.6f}")
        
        for i, result in enumerate(fold_results):
            print(f"Fold {i + 1}: Val Loss = {result['best_val_loss']:.6f}, Train Loss = {result['final_train_loss']:.6f}")
        
        # Save aggregated results
        cv_results = {
            'k_folds': args.k_folds,
            'val_loss_mean': float(np.mean(val_losses_all_folds)),
            'val_loss_std': float(np.std(val_losses_all_folds)),
            'train_loss_mean': float(np.mean(train_losses_all_folds)),
            'train_loss_std': float(np.std(train_losses_all_folds)),
            'fold_results': fold_results
        }
        
        with open(os.path.join(checkpoint_dir, 'cv_results.json'), 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        # Create aggregated plot
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Individual fold loss curves
        plt.subplot(2, 2, 1)
        for i, result in enumerate(fold_results):
            epochs = range(1, len(result['val_losses']) + 1)
            plt.plot(epochs, result['val_losses'], label=f'Fold {i+1}', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss - All Folds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot 2: Box plot of final validation losses
        plt.subplot(2, 2, 2)
        plt.boxplot(val_losses_all_folds)
        plt.ylabel('Best Validation Loss')
        plt.title('Distribution of Best Validation Losses')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Mean validation loss with error bars
        plt.subplot(2, 2, 3)
        max_epochs = max(len(result['val_losses']) for result in fold_results)
        mean_val_losses = []
        std_val_losses = []
        
        for epoch in range(max_epochs):
            epoch_losses = []
            for result in fold_results:
                if epoch < len(result['val_losses']):
                    epoch_losses.append(result['val_losses'][epoch])
            
            if epoch_losses:
                mean_val_losses.append(np.mean(epoch_losses))
                std_val_losses.append(np.std(epoch_losses))
        
        epochs = range(1, len(mean_val_losses) + 1)
        mean_val_losses = np.array(mean_val_losses)
        std_val_losses = np.array(std_val_losses)
        
        plt.plot(epochs, mean_val_losses, 'b-', label='Mean', linewidth=2)
        plt.fill_between(epochs, mean_val_losses - std_val_losses, 
                        mean_val_losses + std_val_losses, alpha=0.3, color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Mean Validation Loss Across Folds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot 4: Final performance comparison
        plt.subplot(2, 2, 4)
        fold_nums = range(1, len(fold_results) + 1)
        plt.bar(fold_nums, val_losses_all_folds, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.axhline(y=np.mean(val_losses_all_folds), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(val_losses_all_folds):.6f}')
        plt.xlabel('Fold')
        plt.ylabel('Best Validation Loss')
        plt.title('Best Validation Loss by Fold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, 'cv_summary.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nCross-validation complete! Results saved to {checkpoint_dir}")
        print(f"Best performing fold: {np.argmin(val_losses_all_folds) + 1} (Val Loss: {np.min(val_losses_all_folds):.6f})")

else:
    # === SINGLE SPLIT TRAINING (ORIGINAL LOGIC) ===
    print("\n=== SINGLE SPLIT TRAINING ===")
    
    # Initialize training loss tracking - combine existing history with new losses
    train_losses = existing_train_losses.copy()  # Start with existing history
    val_losses = existing_val_losses.copy()      # Start with existing history
    new_train_losses = []  # Track only new losses for this session
    new_val_losses = []    # Track only new losses for this session

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        # Append to both complete history and new session tracking
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        new_train_losses.append(train_loss)
        new_val_losses.append(val_loss)
        scheduler.step(val_loss)  # Step the scheduler with validation loss
        print(f"Epoch {epoch}: Train Loss = {train_loss:.2e}, Val Loss = {val_loss:.2e}, LR = {optimizer.param_groups[0]['lr']:.2e}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save only if save_interval epochs have passed or it's the last epoch
            if (epoch - last_best_save_epoch) >= save_interval or epoch == num_epochs:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
                last_best_save_epoch = epoch
                print(f"Saved new best model at epoch {epoch}.")
            else:
                print(f"New best model found, but not saving (interval: {save_interval} epochs).")
        # Always save optimizer state
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pth'))
        # Update config
        config['last_epoch'] = epoch
        config['best_val_loss'] = best_val_loss
        with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    # Save complete training history (preserves all previous epochs + new ones)
    np.save(os.path.join(checkpoint_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(checkpoint_dir, 'val_losses.npy'), np.array(val_losses))

    # Plot complete training history
    plt.figure(figsize=(10, 6))
    total_epochs = len(train_losses)
    epoch_range = range(1, total_epochs + 1)

    plt.plot(epoch_range, train_losses, label='Train Loss', alpha=0.8)
    plt.plot(epoch_range, val_losses, label='Validation Loss', alpha=0.8)

    # Highlight the new training epochs if resuming
    if len(existing_train_losses) > 0:
        new_epoch_start = len(existing_train_losses) + 1
        new_epoch_range = range(new_epoch_start, total_epochs + 1)
        plt.plot(new_epoch_range, new_train_losses, label='Train Loss (This Session)', 
                 linewidth=3, alpha=0.9)
        plt.plot(new_epoch_range, new_val_losses, label='Validation Loss (This Session)', 
                 linewidth=3, alpha=0.9)
        plt.axvline(x=new_epoch_start-0.5, color='red', linestyle='--', alpha=0.7, 
                    label=f'Resumed at Epoch {new_epoch_start}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Complete History: {total_epochs} epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(checkpoint_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Total epochs trained: {total_epochs}")
    if len(existing_train_losses) > 0:
        print(f"Previous epochs: {len(existing_train_losses)}, New epochs this session: {len(new_train_losses)}") 