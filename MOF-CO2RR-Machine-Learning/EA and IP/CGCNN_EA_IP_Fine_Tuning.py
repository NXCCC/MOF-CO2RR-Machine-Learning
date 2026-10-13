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
from data import collate_pool, MultiTargetCIFData
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
    self.mean = torch.mean(tensor, dim=0)
    self.std = torch.std(tensor, dim=0)
  def norm(self, tensor):
    # Keep mean and standard deviation on the same device as the input tensor.
    return (tensor - self.mean.to(tensor.device)) / self.std.to(tensor.device)
  def denorm(self, tensor):
    # Keep mean and standard deviation on the same device as the input tensor.
    return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)
  def state_dict(self):
    return {'mean': self.mean, 'std': self.std}
  def to(self, device):
    """Move the normalizer to the specified device."""
    self.mean = self.mean.to(device)
    self.std = self.std.to(device)
    return self


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
  def __init__(self, patience=20, min_delta=0.0001, verbose=True):
    self.patience = patience
    self.min_delta = min_delta
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.best_epoch = 0
  def __call__(self, val_loss, epoch):
    score = -val_loss
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
# Physics-Constrained Loss
# ==========================================================


class EAIPLoss(nn.Module):
  def __init__(self, lambda_physics=0.5, lambda_gap_separate=0.3):
    super(EAIPLoss, self).__init__()
    self.mae = nn.L1Loss()
    self.lambda_physics = lambda_physics
    self.lambda_gap_separate = lambda_gap_separate
  def forward(self, pred, target):
    mae_loss = self.mae(pred, target)
    pred_ea, pred_ip = pred[:, 0], pred[:, 1]
    true_ea, true_ip = target[:, 0], target[:, 1]
    pred_gap = pred_ea - pred_ip
    true_gap = true_ea - true_ip
    gap_loss = torch.mean(torch.abs(pred_gap - true_gap))
    total_loss = mae_loss + self.lambda_physics * gap_loss + \
          self.lambda_gap_separate * gap_loss
    return total_loss, {
      'mae_loss': mae_loss.item(),
      'gap_loss': gap_loss.item(),
      'total_loss': total_loss.item()
    }
# ==========================================================
# Model Loading
# ==========================================================


def load_model_for_finetuning(checkpoint_path, device, args):
  print(f"\nLoading the pretrained model from {checkpoint_path}...")
  checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
  # Retrieve the model architecture parameters from the checkpoint.
  if 'args' in checkpoint:
    saved_args = checkpoint['args']
    atom_fea_len = saved_args.get('atom_fea_len', args.atom_fea_len)
    n_conv = saved_args.get('n_conv', args.n_conv)
    h_fea_len = saved_args.get('h_fea_len', args.h_fea_len)
    n_h = saved_args.get('n_h', args.n_h)
    print(f"Using saved model parameters: atom_fea_len={atom_fea_len}, n_conv={n_conv}, "
       f"h_fea_len={h_fea_len}, n_h={n_h}")
  else:
    atom_fea_len = args.atom_fea_len
    n_conv = args.n_conv
    h_fea_len = args.h_fea_len
    n_h = args.n_h
    print(f"Using command-line model parameters")
  state_dict = checkpoint['state_dict']
  if 'embedding.weight' in state_dict:
    orig_atom_fea_len = state_dict['embedding.weight'].shape[1]
    print(f"✓ Detected from checkpoint: orig_atom_fea_len = {orig_atom_fea_len}")
  else:
    orig_atom_fea_len = 92  # Default value
    print(f"⚠ Using default orig_atom_fea_len = {orig_atom_fea_len}")
  if 'convs.0.fc_full.weight' in state_dict:
    fc_full_in = state_dict['convs.0.fc_full.weight'].shape[1]
    nbr_fea_len = fc_full_in - 2 * atom_fea_len
    print(f"✓ Detected from checkpoint: nbr_fea_len = {nbr_fea_len}")
  else:
    nbr_fea_len = 41  # Default value
    print(f"⚠ Using default nbr_fea_len = {nbr_fea_len}")
  model = CrystalGraphConvNet(
    orig_atom_fea_len=orig_atom_fea_len,  # Use the detected value
    nbr_fea_len=nbr_fea_len,              # Use the detected value
    atom_fea_len=atom_fea_len,
    n_conv=n_conv,
    h_fea_len=h_fea_len,
    n_h=n_h,
    classification=False
  )
  keys_to_remove = [k for k in state_dict.keys() if 'fc_out' in k]
  for key in keys_to_remove:
    del state_dict[key]
    print(f"Removing original output layer: {key}")
  model.load_state_dict(state_dict, strict=False)
  print("✓ Pretrained weights loaded successfully")
  num_ftrs = model.fc_out.in_features
  model.fc_out = nn.Linear(num_ftrs, 2)
  print(f"✓ Output layer: {num_ftrs} -> 2 (EA, IP)")
  return model.to(device)
# ==========================================================
# Arguments
# ==========================================================
parser = argparse.ArgumentParser(description='EA-IP Fine-tuning')
parser.add_argument('datapath', help='Dataset directory')
parser.add_argument('modelpath', help='Path to the pretrained model')
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--patience', default=30, type=int)
parser.add_argument('--min-delta', default=0.0001, type=float)
parser.add_argument('--output-dir', default='results_EA_IP', type=str)
parser.add_argument('--atom-fea-len', default=64, type=int)
parser.add_argument('--n-conv', default=3, type=int)
parser.add_argument('--h-fea-len', default=128, type=int)
parser.add_argument('--n-h', default=1, type=int)
parser.add_argument('--lambda-physics', default=0.5, type=float)
parser.add_argument('--lambda-gap', default=0.3, type=float)
parser.add_argument('--lr-scheduler', default='reduce',
         choices=['reduce', 'cosine', 'none'])
args = parser.parse_args()
# ==========================================================
# Main
# ==========================================================


def main():
  set_seed(args.seed)
  cuda = (not args.disable_cuda) and torch.cuda.is_available()
  device = torch.device("cuda" if cuda else "cpu")
  os.makedirs(args.output_dir, exist_ok=True)
  writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
  log = open(os.path.join(args.output_dir, "training_log.txt"), "w", encoding="utf-8")
  def log_print(msg):
    print(msg)
    log.write(msg + "\n")
    log.flush()
  log_print(f"==== EA-IP training started @ {time.ctime()} ====")
  log_print(f"Device: {device}")
  history = {
    'epoch': [], 'train_loss': [], 'train_mae_ea': [], 'train_mae_ip': [],
    'train_mae_gap': [], 'val_mae_ea': [], 'val_mae_ip': [], 'val_mae_gap': [],
    'val_total_mae': [], 'test_mae_ea': [], 'test_mae_ip': [], 'test_mae_gap': [],
    'learning_rate': []
  }
  log_print("\nLoading dataset...")
  dataset = MultiTargetCIFData(args.datapath, disable_save_torch=True)
  log_print(f"Dataset size: {len(dataset)} samples")
  sample = dataset[0]
  log_print(f"Target shape: {sample[1].shape}")
  all_indices = list(range(len(dataset)))
  train_idx, test_idx = train_test_split(all_indices, test_size=0.1, random_state=args.seed)
  train_idx, val_idx = train_test_split(train_idx, test_size=0.1111, random_state=args.seed)
  log_print(f"Train: {len(train_idx)} | Validation: {len(val_idx)} | Test: {len(test_idx)}")
  train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
              shuffle=True, collate_fn=collate_pool, num_workers=0, pin_memory=cuda)
  val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
             shuffle=False, collate_fn=collate_pool, num_workers=0, pin_memory=cuda)
  test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size,
              shuffle=False, collate_fn=collate_pool, num_workers=0, pin_memory=cuda)
  train_targets = torch.stack([dataset[i][1] for i in train_idx])
  normalizer = Normalizer(train_targets)
  normalizer.to(device)  # Move the normalizer to the selected device.
  log_print(f"EA: mean={normalizer.mean[0]:.3f}, std={normalizer.std[0]:.3f}")
  log_print(f"IP: mean={normalizer.mean[1]:.3f}, std={normalizer.std[1]:.3f}")
  model = load_model_for_finetuning(args.modelpath, device, args)
  criterion = EAIPLoss(lambda_physics=args.lambda_physics,
            lambda_gap_separate=args.lambda_gap)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  if args.lr_scheduler == 'reduce':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.5, patience=10, verbose=True)
  elif args.lr_scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=args.epochs, eta_min=1e-6)
  else:
    scheduler = None
  early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
  best_val_mae = float('inf')
  start_time = time.time()
  log_print("\n" + "="*70)
  log_print("Training started")
  log_print("="*70)
  for epoch in range(args.epochs):
    epoch_start = time.time()
    train_loss, train_metrics = train(train_loader, model, optimizer,
                    criterion, normalizer, device)
    val_metrics = evaluate(val_loader, model, normalizer, device)
    test_metrics = evaluate(test_loader, model, normalizer, device)
    current_lr = optimizer.param_groups[0]['lr']
    if scheduler is not None:
      if args.lr_scheduler == 'reduce':
        scheduler.step(val_metrics['total_mae'])
      else:
        scheduler.step()
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_mae_ea'].append(train_metrics['mae_ea'])
    history['train_mae_ip'].append(train_metrics['mae_ip'])
    history['train_mae_gap'].append(train_metrics['mae_gap'])
    history['val_mae_ea'].append(val_metrics['mae_ea'])
    history['val_mae_ip'].append(val_metrics['mae_ip'])
    history['val_mae_gap'].append(val_metrics['mae_gap'])
    history['val_total_mae'].append(val_metrics['total_mae'])
    history['test_mae_ea'].append(test_metrics['mae_ea'])
    history['test_mae_ip'].append(test_metrics['mae_ip'])
    history['test_mae_gap'].append(test_metrics['mae_gap'])
    history['learning_rate'].append(current_lr)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("MAE/val_EA", val_metrics['mae_ea'], epoch)
    writer.add_scalar("MAE/val_IP", val_metrics['mae_ip'], epoch)
    writer.add_scalar("LearningRate", current_lr, epoch)
    is_best = val_metrics['total_mae'] < best_val_mae
    if is_best:
      best_val_mae = val_metrics['total_mae']
    epoch_time = time.time() - epoch_start
    log_print(
      f"Epoch {epoch:03d} [{epoch_time:.1f}s] | "
      f"Loss {train_loss:.4f} | "
      f"Val EA:{val_metrics['mae_ea']:.4f} IP:{val_metrics['mae_ip']:.4f} Gap:{val_metrics['mae_gap']:.4f} "
      f"{'★' if is_best else ''}"
    )
    save_checkpoint({
      'epoch': epoch,
      'args': vars(args),
      'state_dict': model.state_dict(),
      'normalizer': normalizer.state_dict(),
      'best_val_mae': best_val_mae,
      'history': history
    }, is_best, args.output_dir)
    if (epoch + 1) % 10 == 0:
      save_history(history, args.output_dir)
    if early_stopping(val_metrics['total_mae'], epoch):
      log_print(f"\nEarly stopping triggered. Best epoch: {early_stopping.best_epoch}")
      break
  log_print(f"\nTraining completed. Elapsed time: {(time.time()-start_time)/60:.1f} minutes")
  best_checkpoint = torch.load(os.path.join(args.output_dir, 'model_best.pth.tar'),
                map_location=device, weights_only=False)
  model.load_state_dict(best_checkpoint['state_dict'])
  final_test = evaluate(test_loader, model, normalizer, device)
  log_print("\n==== Final test results ====")
  log_print(f"EA MAE  = {final_test['mae_ea']:.4f} eV")
  log_print(f"IP MAE  = {final_test['mae_ip']:.4f} eV")
  log_print(f"Gap MAE = {final_test['mae_gap']:.4f} eV")
  log_print(f"EA R²   = {final_test['r2_ea']:.4f}")
  log_print(f"IP R²   = {final_test['r2_ip']:.4f}")
  save_predictions(test_loader, model, normalizer, device,
          os.path.join(args.output_dir, 'test_predictions.csv'))
  save_history(history, args.output_dir)
  log.close()
  writer.close()
  print("\n✅ Completed!")


def train(loader, model, optimizer, criterion, normalizer, device):
  model.train()
  losses, mae_ea, mae_ip, mae_gap = [AverageMeter() for _ in range(4)]
  for input_data, target, _ in loader:
    target = target.to(device)
    target_norm = normalizer.norm(target)
    input_var = (input_data[0].to(device), input_data[1].to(device),
          input_data[2].to(device), [i.to(device) for i in input_data[3]])
    pred_norm = model(*input_var)
    loss, _ = criterion(pred_norm, target_norm)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred = normalizer.denorm(pred_norm)
    losses.update(loss.item(), target.size(0))
    mae_ea.update(torch.mean(torch.abs(pred[:, 0] - target[:, 0])).item(), target.size(0))
    mae_ip.update(torch.mean(torch.abs(pred[:, 1] - target[:, 1])).item(), target.size(0))
    mae_gap.update(torch.mean(torch.abs((pred[:, 0] - pred[:, 1]) -
                     (target[:, 0] - target[:, 1]))).item(), target.size(0))
  return losses.avg, {'mae_ea': mae_ea.avg, 'mae_ip': mae_ip.avg, 'mae_gap': mae_gap.avg}


def evaluate(loader, model, normalizer, device):
  model.eval()
  y_true_ea, y_pred_ea, y_true_ip, y_pred_ip = [], [], [], []
  with torch.no_grad():
    for input_data, target, _ in loader:
      input_var = (input_data[0].to(device), input_data[1].to(device),
            input_data[2].to(device), [i.to(device) for i in input_data[3]])
      pred_norm = model(*input_var)
      pred = normalizer.denorm(pred_norm).cpu().numpy()
      target_np = target.numpy()
      y_pred_ea.append(pred[:, 0])
      y_pred_ip.append(pred[:, 1])
      y_true_ea.append(target_np[:, 0])
      y_true_ip.append(target_np[:, 1])
  y_true_ea = np.concatenate(y_true_ea)
  y_pred_ea = np.concatenate(y_pred_ea)
  y_true_ip = np.concatenate(y_true_ip)
  y_pred_ip = np.concatenate(y_pred_ip)
  return {
    'mae_ea': np.mean(np.abs(y_true_ea - y_pred_ea)),
    'mae_ip': np.mean(np.abs(y_true_ip - y_pred_ip)),
    'mae_gap': np.mean(np.abs((y_true_ea - y_true_ip) - (y_pred_ea - y_pred_ip))),
    'total_mae': (np.mean(np.abs(y_true_ea - y_pred_ea)) +
          np.mean(np.abs(y_true_ip - y_pred_ip))) / 2,
    'r2_ea': r2_score(y_true_ea, y_pred_ea),
    'r2_ip': r2_score(y_true_ip, y_pred_ip)
  }


def save_checkpoint(state, is_best, output_dir):
  torch.save(state, os.path.join(output_dir, 'checkpoint.pth.tar'))
  if is_best:
    shutil.copyfile(os.path.join(output_dir, 'checkpoint.pth.tar'),
           os.path.join(output_dir, 'model_best.pth.tar'))


def save_history(history, output_dir):
  pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)


def save_predictions(loader, model, normalizer, device, save_path):
  model.eval()
  results = []
  with torch.no_grad():
    for input_data, target, cif_ids in loader:
      input_var = (input_data[0].to(device), input_data[1].to(device),
            input_data[2].to(device), [i.to(device) for i in input_data[3]])
      pred = normalizer.denorm(model(*input_var)).cpu().numpy()
      target_np = target.numpy()
      for i in range(len(cif_ids)):
        results.append({
          'material_id': cif_ids[i],
          'true_EA': target_np[i, 0], 'pred_EA': pred[i, 0],
          'true_IP': target_np[i, 1], 'pred_IP': pred[i, 1],
          'true_gap': target_np[i, 0] - target_np[i, 1],
          'pred_gap': pred[i, 0] - pred[i, 1],
          'error_EA': abs(target_np[i, 0] - pred[i, 0]),
          'error_IP': abs(target_np[i, 1] - pred[i, 1])
        })
  pd.DataFrame(results).to_csv(save_path, index=False)
  print(f"Predictions saved to: {save_path}")


if __name__ == '__main__':
  main()
