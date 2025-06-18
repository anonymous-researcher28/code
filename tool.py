#!/usr/bin/env python

import os
import csv
import math
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve, roc_auc_score

from dataset import MoleculeRecord, MoleculeDataset
from model import FinH2AN

def ensure_dir(path, is_directory=True):
    if not is_directory:
        path = os.path.dirname(path)
    if path:
        os.makedirs(path, exist_ok=True)

def init_logger(name, log_dir, seed):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    ensure_dir(log_dir)
    file_handler = logging.FileHandler(os.path.join(log_dir, f'debug_{seed}.log'))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger

def load_csv_data(file_path, config):
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        lines = list(reader)
    records = [MoleculeRecord(line, config) for line in lines]
    dataset = MoleculeDataset(records)
    valid_records = [rec for rec in dataset.records if rec.mol is not None]
    print(f"Loaded {len(valid_records)} valid molecules (from {len(records)} entries).")
    return MoleculeDataset(valid_records)

def scale_targets(dataset):
    targets = np.array(dataset.get_targets(), dtype=float)
    mean_vals = np.nanmean(targets, axis=0)
    mean_vals = np.where(np.isnan(mean_vals), np.zeros(mean_vals.shape), mean_vals)
    std_vals = np.nanstd(targets, axis=0)
    std_vals = np.where((np.isnan(std_vals)) | (std_vals == 0), np.ones(std_vals.shape), std_vals)
    scaled = (targets - mean_vals) / std_vals
    scaled = np.where(np.isnan(scaled), None, scaled)
    dataset.update_all_targets(scaled.tolist())
    return mean_vals, std_vals

def get_loss_function(task_type):
    if task_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')
    elif task_type == 'regression':
        return nn.MSELoss(reduction='none')
    else:
        raise ValueError("Unknown task type.")

def prc_auc(true_labels, predictions):
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    return auc(recall, precision)

def rmse(true_labels, predictions):
    return math.sqrt(mean_squared_error(true_labels, predictions))

def fetch_metric(metric_name):
    metrics = {
        'auc': roc_auc_score,
        'prc-auc': prc_auc,
        'rmse': rmse,
    }
    if metric_name in metrics:
        return metrics[metric_name]
    else:
        raise ValueError("Unknown metric.")

def save_checkpoint(file_path, model, scaler, config):
    state = {
        'config': config,
        'model_state': model.state_dict(),
        'scaler': {'mean': scaler[0], 'std': scaler[1]} if scaler is not None else None,
    }
    torch.save(state, file_path)

def load_checkpoint(file_path, use_cuda, logger=None, extra_args=None):
    debug = logger.debug if logger else print
    state = torch.load(file_path, map_location=lambda storage, loc: storage)
    config = state['config']
    if extra_args:
        for k, v in vars(extra_args).items():
            if not hasattr(config, k):
                setattr(config, k, v)
    model = FinH2AN(config)
    model_state = model.state_dict()
    load_dict = {}
    for key, val in state['model_state'].items():
        if key not in model_state:
            debug(f"Parameter {key} not found in model.")
        elif model_state[key].shape != val.shape:
            debug(f"Shape mismatch for parameter {key}.")
        else:
            load_dict[key] = val
            debug(f"Loaded parameter {key}.")
    model_state.update(load_dict)
    model.load_state_dict(model_state)
    if use_cuda:
        model = model.to(torch.device("cuda"))
    return model

def get_saved_scaler(file_path):
    state = torch.load(file_path, map_location=lambda storage, loc: storage)
    if state['scaler'] is not None:
        return state['scaler']['mean'], state['scaler']['std']
    else:
        return None

def load_saved_config(file_path):
    state = torch.load(file_path, map_location=lambda storage, loc: storage)
    return state['config']

class NoamLearningRate(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch, init_lr, peak_lr, final_lr):
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == len(peak_lr) == len(final_lr)
        self.num_groups = len(optimizer.param_groups)
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.peak_lr = np.array(peak_lr)
        self.final_lr = np.array(final_lr)
        self.current_step = 0
        self.lrs = list(init_lr)
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.lr_increments = (self.peak_lr - self.init_lr) / self.warmup_steps
        self.gamma = (self.final_lr / self.peak_lr) ** (1 / (self.total_steps - self.warmup_steps))
        super(NoamLearningRate, self).__init__(optimizer)
        
    def get_lr(self):
        return list(self.lrs)
    
    def step(self, current_step=None):
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1
        for i in range(self.num_groups):
            if self.current_step <= self.warmup_steps[i]:
                self.lrs[i] = self.init_lr[i] + self.current_step * self.lr_increments[i]
            elif self.current_step <= self.total_steps[i]:
                self.lrs[i] = self.peak_lr[i] * (self.gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:
                self.lrs[i] = self.final_lr[i]
            self.optimizer.param_groups[i]['lr'] = self.lrs[i]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_data_file', type=str, required=True, help="Path to input CSV file.")
    parser.add_argument('--hp_output_model', type=str, default='model_checkpoint', help="Path to save model checkpoint.")
    parser.add_argument('--hp_log_dir', type=str, default='logs', help="Directory for logs.")
    parser.add_argument('--hp_data_type', type=str, choices=['classification', 'regression'], required=True, help="Dataset type.")
    parser.add_argument('--split_method', type=str, choices=['random', 'scaffold'], default='random', help="Data splitting method.")
    parser.add_argument('--split_ratios', type=float, nargs=3, default=[0.8, 0.1, 0.1], help="Train, validation, test ratios.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--hp_hidden_dim', type=int, default=256, help="Hidden dimension for model layers.")
    parser.add_argument('--hp_dropout', type=float, default=0.5, help="Dropout for FFN and attention blocks.")
    parser.add_argument('--hp_output_dim', type=int, default=1, help="Number of output tasks.")
    parser.add_argument('--hp_mol_e_dim', type=int, default=128, help="MolHyGAN projection dimension for hyperedges.")
    parser.add_argument('--hp_mol_q_dim', type=int, default=32, help="MolHyGAN query dimension.")
    parser.add_argument('--hp_mol_v_dim', type=int, default=64, help="MolHyGAN value dimension.")
    parser.add_argument('--hp_fp_proj_dim', type=int, default=256, help="Projection dimension in fingerprint branch.")
    parser.add_argument('--init_lr', type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument('--max_lr', type=float, default=1e-3, help="Peak learning rate.")
    parser.add_argument('--final_lr', type=float, default=1e-4, help="Final learning rate.")
    parser.add_argument('--warmup_epochs', type=float, default=2.0, help="Number of warmup epochs for LR scheduler.")
    parser.add_argument('--num_lrs', type=int, default=1, help="Number of learning rate groups.")
    parser.add_argument('--patience', type=int, default=7, help="Early stopping patience.")
    parser.add_argument('--metric', type=str, choices=['auc', 'prc-auc', 'rmse'], required=True, help="Evaluation metric.")
    parser.add_argument('--hp_cuda', action='store_true', help="Use CUDA if available.")
    parser.add_argument('--he_dim', default=384, type=int,  help="Hyperedge dimension for HFPN.")

    args = parser.parse_args()
    if torch.cuda.is_available():
        args.hp_cuda = True
        print("run cuda")
    return args
