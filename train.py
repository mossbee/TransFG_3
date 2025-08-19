# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta

import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

# Removed reduce_mean function as it's no longer needed without distributed training

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """Load checkpoint and return training state"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model'])
    
    # Load training states if available
    global_step = checkpoint.get('global_step', 0)
    best_acc = checkpoint.get('best_acc', 0)
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Loaded optimizer state")
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded scheduler state")
    
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        logger.info("Loaded scaler state")
    
    logger.info(f"Resuming from global_step: {global_step}, best_acc: {best_acc}")
    return global_step, best_acc

def save_model(args, model, optimizer=None, scheduler=None, scaler=None, global_step=0, best_acc=0):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    checkpoint = {
        'model': model_to_save.state_dict(),
        'global_step': global_step,
        'best_acc': best_acc,
        'args': args,
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    if args.fp16 and scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
    elif args.dataset == "nd_twin":
        num_classes = 347

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,                                                   smoothing_value=args.smoothing_value)

    model.load_from(np.load(args.pretrained_dir))
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def calculate_verification_metrics(distances, labels):
    """
    Calculate verification metrics from cosine distances and labels
    """
    # Convert to numpy if needed
    if torch.is_tensor(distances):
        distances = distances.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Calculate AUC
    auc = roc_auc_score(labels, -distances)  # negative because smaller distance = higher similarity
    
    # Calculate EER
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    # Calculate verification accuracy at EER threshold
    eer_threshold_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    eer_threshold = thresholds[eer_threshold_idx]
    predictions = (-distances >= eer_threshold).astype(int)
    verification_accuracy = np.mean(predictions == labels)
    
    return {
        'auc': auc,
        'eer': eer,
        'verification_accuracy': verification_accuracy,
        'eer_threshold': -eer_threshold  # Convert back to similarity threshold
    }

def valid_verify(args, model, writer, test_loader, global_step):
    """
    Validation for verification task using cosine similarity
    """
    logger.info("***** Running Verification Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_distances = []
    all_labels = []
    
    epoch_iterator = tqdm(test_loader,
                          desc="Validating Verification...",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)
    
    for step, batch in enumerate(epoch_iterator):
        img1, img2, labels = batch
        img1, img2, labels = img1.to(args.device), img2.to(args.device), labels.to(args.device)
        
        with torch.no_grad():
            # Get features for both images
            features1 = model(img1, return_features=True)
            features2 = model(img2, return_features=True)
            
            # Calculate cosine similarity
            features1_norm = F.normalize(features1, p=2, dim=1)
            features2_norm = F.normalize(features2, p=2, dim=1)
            cosine_sim = torch.sum(features1_norm * features2_norm, dim=1)
            cosine_distance = 1 - cosine_sim  # Convert similarity to distance
            
            all_distances.append(cosine_distance.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all results
    all_distances = torch.cat(all_distances)
    all_labels = torch.cat(all_labels)
    
    # Calculate metrics
    metrics = calculate_verification_metrics(all_distances, all_labels)
    
    logger.info("\n")
    logger.info("Verification Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("AUC: %2.5f" % metrics['auc'])
    logger.info("EER: %2.5f" % metrics['eer'])
    logger.info("Verification Accuracy: %2.5f" % metrics['verification_accuracy'])
    logger.info("EER Threshold: %2.5f" % metrics['eer_threshold'])
    
    # Log to tensorboard
    writer.add_scalar("verification/auc", scalar_value=metrics['auc'], global_step=global_step)
    writer.add_scalar("verification/eer", scalar_value=metrics['eer'], global_step=global_step)
    writer.add_scalar("verification/accuracy", scalar_value=metrics['verification_accuracy'], global_step=global_step)
    
    return metrics['auc']  # Return AUC as the main metric for model selection

def valid_classification(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            # Use autocast for mixed precision in validation
            with autocast(enabled=args.fp16):
                logits = model(x)
                eval_loss = loss_fct(logits, y)
                eval_loss = eval_loss.mean()
            
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    val_accuracy = accuracy

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)
    writer.add_scalar("test/accuracy", scalar_value=val_accuracy, global_step=global_step)
        
    return val_accuracy

def train(args, model):
    """ Train the model """
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    is_verification = hasattr(test_loader.dataset, 'pairs')

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler() if args.fp16 else None

    # Initialize training state
    global_step, best_acc = 0, 0
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        global_step, best_acc = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        logger.info(f"Resumed training from step {global_step} with best_acc {best_acc}")

    # Removed distributed training setup

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Starting from global step = %d", global_step)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    start_time = time.time()
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=False)
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch

            # Use autocast for mixed precision
            with autocast(enabled=args.fp16):
                loss, logits = model(x, y)
                loss = loss.mean()

            preds = torch.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            # Scale loss for mixed precision training
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                
                # Gradient clipping and optimizer step with mixed precision
                if args.fp16:
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0:
                    with torch.no_grad():
                        if is_verification:
                            accuracy = valid_verify(args, model, writer, test_loader, global_step)
                            logger.info("Current AUC: %f" % accuracy)
                        else:
                            accuracy = valid_classification(args, model, writer, test_loader, global_step)
                            logger.info("Current Classification Accuracy: %f" % accuracy)
                    
                    if best_acc < accuracy:
                        save_model(args, model, optimizer, scheduler, scaler, global_step, accuracy)
                        best_acc = accuracy
                    
                    if is_verification:
                        logger.info("best AUC so far: %f" % best_acc)
                    else:
                        logger.info("best accuracy so far: %f" % best_acc)
                    model.train()

                if global_step >= t_total:
                    break
        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        train_accuracy = accuracy
        logger.info("train accuracy so far: %f" % train_accuracy)
        losses.reset()
        if global_step >= t_total:
            break

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017", "nd_twin"], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/opt/tiger/minist')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/opt/tiger/minist/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--resume", type=str, default=None,
                        help="resume training from checkpoint")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--eval_mode", type=str, default="verification", choices=["classification", "verification"],
                        help="Evaluation mode.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    # Removed local_rank argument as distributed training is no longer supported
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    # Removed fp16_opt_level argument as PyTorch native AMP doesn't need optimization levels
    # Removed loss_scale argument as PyTorch native AMP handles scaling automatically

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    args = parser.parse_args()

    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s, 16-bits training: %s" %
                   (args.device, args.n_gpu, args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    # Training
    train(args, model)

if __name__ == "__main__":
    main()
