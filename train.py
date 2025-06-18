

import os
import numpy as np
import torch
from torch.optim import Adam, AdamW
from copy import deepcopy
import random
import copy

from tool import (
    init_logger, load_csv_data, scale_targets, fetch_metric,
    get_loss_function, save_checkpoint, load_checkpoint, NoamLearningRate, parse_arguments
)
from dataset import MoleculeDataset, split_dataset
from model import FinH2AN
from torch.optim.lr_scheduler import ExponentialLR


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(42)


def create_single_task_dataset(dataset, task_index):
    new_records = []
    for rec in dataset.records:
        new_rec = copy.deepcopy(rec)
        new_rec.targets = [rec.targets[task_index]]
        new_records.append(new_rec)
    return MoleculeDataset(new_records)


def train_epoch(model, data, loss_fn, optimizer, scheduler, args):
    model.train()
    data.shuffle(args.seed)
    batch_step = args.batch_size
    for i in range(0, len(data), batch_step):
        if i + batch_step > len(data):
            break
        batch_data = MoleculeDataset(data.records[i:i+batch_step])
        smiles = batch_data.get_smiles()
        labels = batch_data.get_targets()

        mask = torch.Tensor([[x is not None for x in task] for task in labels])
        targets = torch.Tensor([[0 if x is None else x for x in task] for task in labels])
        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()
        weight = torch.ones(targets.shape)
        if args.hp_cuda:
            weight = weight.cuda()

        model.zero_grad()
        predictions = model(smiles)
        loss = loss_fn(predictions, targets) * weight * mask
        loss = loss.sum() / mask.sum()
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLearningRate):
            scheduler.step()
    if isinstance(scheduler, ExponentialLR):
        scheduler.step()


def predict_model(model, data, batch_size, scaler):
    model.eval()
    predictions_all = []
    for i in range(0, len(data), batch_size):
        batch_data = MoleculeDataset(data.records[i:i+batch_size])
        smiles = batch_data.get_smiles()
        with torch.no_grad():
            preds = model(smiles)
        preds = preds.data.cpu().numpy()
        if scaler is not None:
            ave, std = scaler
            preds = preds.astype(float) * std + ave
            preds = np.where(np.isnan(preds), None, preds)
        predictions_all.extend(preds.tolist())
    return predictions_all


def compute_score(predictions, labels, metric_fn, args, logger):
    task_num = args.hp_output_dim
    data_type = args.hp_data_type
    if len(predictions) == 0:
        return [0.5] * task_num if data_type == 'classification' else [float('nan')] * task_num

    pred_by_task = [[] for _ in range(task_num)]
    label_by_task = [[] for _ in range(task_num)]
    for j in range(len(predictions)):
        for i in range(task_num):
            if labels[j][i] is not None:
                pred_by_task[i].append(predictions[j][i])
                label_by_task[i].append(labels[j][i])
    results = []
    for i in range(task_num):
        if data_type == 'classification':
            if all(x == 0 for x in label_by_task[i]) or all(x == 1 for x in label_by_task[i]):
                logger.info('Warning: All labels are 0 or 1. Returning default AUC=0.5.')
                results.append(0.5)
                continue
            if all(x == 0 for x in pred_by_task[i]) or all(x == 1 for x in pred_by_task[i]):
                logger.info('Warning: All predictions are 0 or 1. Returning default AUC=0.5.')
                results.append(0.5)
                continue
        score = metric_fn(label_by_task[i], pred_by_task[i])
        results.append(score)
    return results


def train_single_task(args, logger, train_data, val_data, test_data):
    args.train_data_size = len(train_data)
    scaler = scale_targets(train_data) if args.hp_data_type == 'regression' else None
    loss_fn = get_loss_function(args.hp_data_type)
    metric_fn = fetch_metric(args.metric)
    logger.info('Initializing model...')
    model = FinH2AN(args)
    logger.debug(model)
    if args.hp_cuda:
        model = model.to(torch.device("cuda"))
    save_checkpoint(os.path.join(args.hp_output_model, f'model_{args.seed}.pt'), model, scaler, args)
    optimizer = Adam(model.parameters(), lr=args.init_lr, weight_decay=1e-4) 

    scheduler = NoamLearningRate(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=[args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        peak_lr=[args.max_lr],
        final_lr=[args.final_lr]
    )
    best_score = -float('inf') if args.hp_data_type == 'classification' else float('inf')
    best_epoch = 0
    patience = args.patience
    wait = 0

    data = load_csv_data(args.hp_data_file, args)
    num_tasks = data.num_tasks()
    if not args.task_names or len(args.task_names) != num_tasks:
        args.task_names = [f"Class {i+1}" for i in range(num_tasks)]

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch}")
        train_epoch(model, train_data, loss_fn, optimizer, scheduler, args)
        train_preds = predict_model(model, train_data, args.batch_size, scaler)
        train_labels = train_data.get_targets()
        train_score = compute_score(train_preds, train_labels, metric_fn, args, logger)
        val_preds = predict_model(model, val_data, args.batch_size, scaler)
        val_labels = val_data.get_targets()
        val_score = compute_score(val_preds, val_labels, metric_fn, args, logger)

        for i, task_name in enumerate(args.task_names):
            logger.info(f"{task_name}: Train {args.metric} = {train_score[i]:.6f}")
            logger.info(f"{task_name}: Validation {args.metric} = {val_score[i]:.6f}")
        mean_train = np.nanmean(train_score)
        mean_val = np.nanmean(val_score)
        logger.info(f"Mean Train {args.metric} = {mean_train:.6f}")
        logger.info(f"Mean Validation {args.metric} = {mean_val:.6f}")

        improved = False
        if args.hp_data_type == 'classification' and mean_val > best_score:
            best_score = mean_val
            best_epoch = epoch
            save_checkpoint(os.path.join(args.hp_output_model, f'model_{args.seed}.pt'), model, scaler, args)
            improved = True
        elif args.hp_data_type == 'regression' and mean_val < best_score:
            best_score = mean_val
            best_epoch = epoch
            save_checkpoint(os.path.join(args.hp_output_model, f'model_{args.seed}.pt'), model, scaler, args)
            improved = True
        if improved:
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    logger.info(f"Best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}")
    model = load_checkpoint(os.path.join(args.hp_output_model, f'model_{args.seed}.pt'), args.hp_cuda, logger)
    test_preds = predict_model(model, test_data, args.batch_size, scaler)
    test_score = compute_score(test_preds, test_data.get_targets(), metric_fn, args, logger)
    for i, task_name in enumerate(args.task_names):
        logger.info(f"{task_name}: Test {args.metric} = {test_score[i]:.6f}")
    mean_test = np.nanmean(test_score)
    logger.info(f"Mean Test {args.metric} = {mean_test:.6f}")
    return test_score


def fold_training(args, logger):
    logger.debug('Loading data...')
    data = load_csv_data(args.hp_data_file, args)
    num_tasks = data.num_tasks()
    args.hp_output_dim = num_tasks  
    if not hasattr(args, 'task_names') or not args.task_names or len(args.task_names) != num_tasks:
        args.task_names = [f"Class {i+1}" for i in range(num_tasks)]
    train_data, val_data, test_data = split_dataset(data, args.split_method, args.split_ratios, args.seed, logger)
    test_score = train_single_task(args, logger, train_data, val_data, test_data)
    logger.info("Final Test Metrics:")
    for i, task_name in enumerate(args.task_names):
        logger.info(f"{task_name}: Test {args.metric} = {test_score[i]:.6f}")
    mean_score = np.nanmean(test_score)
    logger.info(f"Mean Test {args.metric} across tasks = {mean_score:.6f}")

    return test_score


def main_train():
    args = parse_arguments()
    
    logger = init_logger("train", args.hp_log_dir, args.seed )
    fold_training(args, logger)


if __name__ == '__main__':
    main_train()
