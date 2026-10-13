import argparse
import os
import shutil
import time
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from data import CIFData, collate_pool
from model import CrystalGraphConvNet

# ==========================================================
# Utils

# ==========================================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Normalizer(object):

    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, tensor):
        return tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping"""

    def __init__(self, patience=20, min_delta=0.0001, verbose=True):
        """
        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum improvement threshold
            verbose: Whether to print progress messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_mae, epoch):
        score = -val_mae  # Negative sign because MAE is minimized
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        return self.early_stop

# ==========================================================
# Args

# ==========================================================
parser = argparse.ArgumentParser(description='CGCNN PBE → HSE Fine-tuning')
parser.add_argument('datapath', help='path to HSE CIFs + id_prop.csv')
parser.add_argument('modelpath', help='pretrained PBE model (.pth.tar)')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--patience', default=20, type=int, help='early stopping patience')
parser.add_argument('--min-delta', default=0.0001, type=float, help='minimum improvement')
parser.add_argument('--output-dir', default='results', type=str, help='output directory')
# Model architecture parameters (consistent with the prediction script)
parser.add_argument('--atom-fea-len', default=64, type=int, help='atom feature length')
parser.add_argument('--n-conv', default=5, type=int, help='number of conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, help='hidden feature length')
parser.add_argument('--n-h', default=1, type=int, help='number of hidden layers')
args = parser.parse_args()

# ==========================================================
# Main

# ==========================================================


def main():
    set_seed(args.seed)
    cuda = (not args.disable_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    log = open(os.path.join(args.output_dir, "training_log.txt"), "w", encoding="utf-8")

    def log_print(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()
    log_print(f"==== Job start @ {time.ctime()} ====")
    log_print(f"Device: {device}")
    log_print(f"Arguments: {args}")

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------
    history = {
        'epoch': [],
        'train_loss': [],
        'train_mae': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': [],
        'test_mae': [],
        'test_rmse': [],
        'test_r2': [],
        'learning_rate': []
    }

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    dataset = CIFData(args.datapath, disable_save_torch=True)
    log_print(f"Dataset size: {len(dataset)}")

    # --------------------------------------------------
    # Manual split (80 / 10 / 10)
    # --------------------------------------------------
    all_indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.1, random_state=args.seed
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.1111, random_state=args.seed
    )
    log_print(f"Train size: {len(train_idx)}")
    log_print(f"Val size: {len(val_idx)}")
    log_print(f"Test size: {len(test_idx)}")
    train_dataset = Subset(dataset, train_idx)
    val_dataset   = Subset(dataset, val_idx)
    test_dataset  = Subset(dataset, test_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pool,
        num_workers=0,
        pin_memory=cuda
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pool,
        num_workers=0,
        pin_memory=cuda
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pool,
        num_workers=0,
        pin_memory=cuda
    )

    # --------------------------------------------------
    # Normalizer (TRAIN SET ONLY)
    # --------------------------------------------------
    train_targets = torch.tensor([dataset[i][1] for i in train_idx], dtype=torch.float)
    normalizer = Normalizer(train_targets)
    log_print(f"Target mean = {normalizer.mean:.4f}, std = {normalizer.std:.4f}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    # Initialize the model using command-line arguments
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        classification=False
    )
    checkpoint = torch.load(args.modelpath, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    log_print("Loaded pretrained PBE weights")
    model = model.to(device)

    # --------------------------------------------------
    # Freeze backbone (partial finetune)
    # --------------------------------------------------
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        # Freeze the first three convolutional layers and fine-tune the final two layers and fc_out
        if 'conv_layers.0' in name or 'conv_layers.1' in name or 'conv_layers.2' in name:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
            trainable_count += param.numel()
    log_print(f"Frozen parameters: {frozen_count:,}")
    log_print(f"Trainable parameters: {trainable_count:,}")

    # --------------------------------------------------
    # Optimizer: separate learning rates for parameter groups
    # --------------------------------------------------
    fc_params = []
    conv_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc_out' in name:
                fc_params.append(param)
            else:
                conv_params.append(param)
    optimizer = optim.Adam([
        {'params': conv_params, 'lr': args.lr},      # Base learning rate
        {'params': fc_params, 'lr': args.lr*5}       # Use a 5x learning rate for fc_out
    ])
    # Learning-rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    criterion = nn.MSELoss()
    best_val_mae = 1e10

    # --------------------------------------------------
    # Early stopping
    # --------------------------------------------------
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True
    )
    # ==================================================
    # Training loop
    # ==================================================
    log_print("\n" + "="*70)
    log_print("Starting training...")
    log_print("="*70)
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # Train
        train_loss, train_mae = train(
            train_loader, model, optimizer, criterion, normalizer, device
        )
        # Validation
        val_metrics = evaluate(val_loader, model, normalizer, device)
        # Test evaluation at every epoch (for plotting)
        test_metrics = evaluate(test_loader, model, normalizer, device)
        # Learning-rate scheduling
        scheduler.step(val_metrics['mae'])
        current_lr = optimizer.param_groups[0]['lr']
        # Update training history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_r2'].append(val_metrics['r2'])
        history['test_mae'].append(test_metrics['mae'])
        history['test_rmse'].append(test_metrics['rmse'])
        history['test_r2'].append(test_metrics['r2'])
        history['learning_rate'].append(current_lr)
        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("MAE/train", train_mae, epoch)
        writer.add_scalar("MAE/val", val_metrics['mae'], epoch)
        writer.add_scalar("MAE/test", test_metrics['mae'], epoch)
        writer.add_scalar("RMSE/val", val_metrics['rmse'], epoch)
        writer.add_scalar("R2/val", val_metrics['r2'], epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        # Check whether this is the best model
        is_best = val_metrics['mae'] < best_val_mae
        if is_best:
            best_val_mae = val_metrics['mae']
        # Print progress
        epoch_time = time.time() - epoch_start
        log_print(
            f"Epoch {epoch:03d} [{epoch_time:.1f}s] | "
            f"LR {current_lr:.2e} | "
            f"Train MAE {train_mae:.4f} | "
            f"Val MAE {val_metrics['mae']:.4f} | "
            f"Test MAE {test_metrics['mae']:.4f} | "
            f"R2 {val_metrics['r2']:.4f} "
            f"{'🌟' if is_best else ''}"
        )
        save_checkpoint({
            'epoch': epoch,
            'args': vars(args),
            'state_dict': model.state_dict(),
            'normalizer': normalizer.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_mae': best_val_mae,
            'history': history
        }, is_best, args.output_dir)
        # Save training history every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_history(history, args.output_dir)
        # Check early stopping
        if early_stopping(val_metrics['mae'], epoch):
            log_print(f"\nEarly stopping triggered! Best epoch: {early_stopping.best_epoch}")
            log_print(f"Best validation MAE: {best_val_mae:.4f}")
            break

    # --------------------------------------------------
    # Training completed
    # --------------------------------------------------
    total_time = time.time() - start_time
    log_print("\n" + "="*70)
    log_print(f"Training completed! Total time: {total_time/60:.2f} minutes")
    log_print("="*70)

    # --------------------------------------------------
    # Load the best model for final evaluation
    # --------------------------------------------------
    log_print("\nLoad the best model for final evaluation...")
    best_checkpoint = torch.load(
        os.path.join(args.output_dir, 'model_hse_best.pth.tar'),
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(best_checkpoint['state_dict'])
    final_test_metrics = evaluate(test_loader, model, normalizer, device)
    final_val_metrics = evaluate(val_loader, model, normalizer, device)
    log_print("\n==== Final test-set performance ====")
    log_print(f"MAE  = {final_test_metrics['mae']:.4f}")
    log_print(f"RMSE = {final_test_metrics['rmse']:.4f}")
    log_print(f"R²   = {final_test_metrics['r2']:.4f}")
    log_print("\n==== Final validation-set performance ====")
    log_print(f"MAE  = {final_val_metrics['mae']:.4f}")
    log_print(f"RMSE = {final_val_metrics['rmse']:.4f}")
    log_print(f"R²   = {final_val_metrics['r2']:.4f}")

    # --------------------------------------------------
    # Save predictions (for scatter plots)
    # --------------------------------------------------
    log_print("\nSaving predictions...")
    save_predictions(test_loader, model, normalizer, device,
                    os.path.join(args.output_dir, 'test_predictions.csv'))
    save_predictions(val_loader, model, normalizer, device,
                    os.path.join(args.output_dir, 'val_predictions.csv'))

    # --------------------------------------------------
    # Save final training history and configuration
    # --------------------------------------------------
    save_history(history, args.output_dir)
    save_config(args, history, final_test_metrics, final_val_metrics, args.output_dir)
    log.close()
    writer.close()
    log_print("\nAll results have been saved!")

# ==========================================================
# Train / Eval

# ==========================================================


def train(loader, model, optimizer, criterion, normalizer, device):
    model.train()
    losses, maes = AverageMeter(), AverageMeter()
    for input_data, target, _ in loader:
        target = target.to(device)
        target_norm = normalizer.norm(target)
        input_var = (
            input_data[0].to(device),
            input_data[1].to(device),
            input_data[2].to(device),
            [i.to(device) for i in input_data[3]]
        )
        pred_norm = model(*input_var)
        loss = criterion(pred_norm, target_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = normalizer.denorm(pred_norm)
        mae = torch.mean(torch.abs(pred - target))
        losses.update(loss.item(), target.size(0))
        maes.update(mae.item(), target.size(0))
    return losses.avg, maes.avg


def evaluate(loader, model, normalizer, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for input_data, target, _ in loader:
            input_var = (
                input_data[0].to(device),
                input_data[1].to(device),
                input_data[2].to(device),
                [i.to(device) for i in input_data[3]]
            )
            pred_norm = model(*input_var)
            pred = normalizer.denorm(pred_norm).cpu().numpy()
            y_pred.append(pred)
            y_true.append(target.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        'mae': np.mean(np.abs(y_true - y_pred)),
        'rmse': np.sqrt(mse),
        'r2': r2_score(y_true, y_pred)
    }

# ==========================================================
# Save Functions

# ==========================================================


def save_checkpoint(state, is_best, output_dir):
    """Save the checkpoint"""
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(output_dir, 'model_hse_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)


def save_history(history, output_dir):
    """Save training historyas CSV"""
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    print(f"Training history saved to: {os.path.join(output_dir, 'training_history.csv')}")


def save_predictions(loader, model, normalizer, device, save_path):
    """Save predictions"""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for input_data, target, _ in loader:
            input_var = (
                input_data[0].to(device),
                input_data[1].to(device),
                input_data[2].to(device),
                [i.to(device) for i in input_data[3]]
            )
            pred_norm = model(*input_var)
            pred = normalizer.denorm(pred_norm).cpu().numpy()
            y_pred.append(pred)
            y_true.append(target.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    df = pd.DataFrame({
        'true': y_true.flatten(),
        'predicted': y_pred.flatten(),
        'error': np.abs(y_true.flatten() - y_pred.flatten())
    })
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to: {save_path}")


def save_config(args, history, test_metrics, val_metrics, output_dir):
    """Save configuration and final results"""
    # Convert NumPy types to native Python types

    def convert_to_serializable(obj):
        """Recursively convert NumPy types to JSON-serializable values"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    config = {
        'args': vars(args),
        'final_results': {
            'test': convert_to_serializable(test_metrics),
            'val': convert_to_serializable(val_metrics),
            'best_epoch': int(np.argmin(history['val_mae'])),
            'best_val_mae': float(np.min(history['val_mae'])),
            'total_epochs': len(history['epoch'])
        }
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {os.path.join(output_dir, 'config.json')}")
if __name__ == '__main__':
    main()
