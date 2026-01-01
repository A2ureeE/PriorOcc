# Copyright (c) PriorOcc. All rights reserved.
"""
Custom training hooks for enhanced training monitoring and checkpointing.
"""
import os
import os.path as osp
import json
from collections import OrderedDict

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class BestCheckpointHook(Hook):
    """
    Hook to save the best checkpoint based on mIoU metric.
    
    Automatically tracks validation mIoU and saves the best model
    to 'best_miou.pth' in the work directory.
    
    Args:
        save_file (str): Filename for best checkpoint. Default: 'best_miou.pth'
        metric_key (str): Key to look for in evaluation results. Default: 'mIoU'
    """
    
    def __init__(self, save_file='best_miou.pth', metric_key='mIoU'):
        self.save_file = save_file
        self.metric_key = metric_key
        self.best_miou = 0.0
        self.best_epoch = -1
    
    def after_val_epoch(self, runner):
        """Called after validation epoch to check and save best model."""
        # Try to get mIoU from evaluation results
        if not hasattr(runner, 'eval_results'):
            return
        
        eval_results = runner.eval_results
        if eval_results is None:
            return
        
        # Look for mIoU in results
        miou = None
        if isinstance(eval_results, dict):
            if self.metric_key in eval_results:
                miou = eval_results[self.metric_key]
            elif 'mIoU' in eval_results:
                miou = eval_results['mIoU']
        
        if miou is None:
            return
        
        # Save if better than current best
        if miou > self.best_miou:
            self.best_miou = miou
            self.best_epoch = runner.epoch + 1
            
            # Save checkpoint
            save_path = osp.join(runner.work_dir, self.save_file)
            runner.save_checkpoint(runner.work_dir, filename_tmpl=self.save_file, save_optimizer=False)
            
            runner.logger.info(
                f'[BestCheckpointHook] New best mIoU: {miou:.4f} at epoch {self.best_epoch}, '
                f'saved to {save_path}'
            )
    
    def after_run(self, runner):
        """Print best result summary after training."""
        runner.logger.info(
            f'[BestCheckpointHook] Best mIoU: {self.best_miou:.4f} at epoch {self.best_epoch}'
        )


@HOOKS.register_module()
class LossCurveHook(Hook):
    """
    Hook to record and plot training/validation loss curves.
    
    Saves loss history to JSON and plots curves as PNG after training.
    
    Args:
        out_dir (str): Output directory. If None, uses runner.work_dir.
        plot_filename (str): Filename for the loss curve plot.
    """
    
    def __init__(self, out_dir=None, plot_filename='loss_curve.png'):
        self.out_dir = out_dir
        self.plot_filename = plot_filename
        self.train_losses = []  # List of (epoch, loss) tuples
        self.val_losses = []    # List of (epoch, loss) tuples
        self.current_epoch_losses = []
    
    def before_run(self, runner):
        """Initialize output directory."""
        if self.out_dir is None:
            self.out_dir = runner.work_dir
    
    def after_train_iter(self, runner):
        """Record training loss after each iteration."""
        # Get loss from log buffer
        if hasattr(runner, 'log_buffer') and runner.log_buffer.ready:
            log_dict = runner.log_buffer.output
            if 'loss' in log_dict:
                self.current_epoch_losses.append(log_dict['loss'])
    
    def after_train_epoch(self, runner):
        """Average training loss for the epoch."""
        if self.current_epoch_losses:
            avg_loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
            self.train_losses.append((runner.epoch + 1, avg_loss))
            self.current_epoch_losses = []
    
    def after_val_epoch(self, runner):
        """Record validation loss after each validation epoch."""
        # Try to get validation loss
        if hasattr(runner, 'log_buffer') and runner.log_buffer.ready:
            log_dict = runner.log_buffer.output
            val_loss = None
            
            # Look for loss in various possible keys
            for key in ['loss', 'val_loss', 'loss_occ']:
                if key in log_dict:
                    val_loss = log_dict[key]
                    break
            
            if val_loss is not None:
                self.val_losses.append((runner.epoch + 1, val_loss))
    
    def after_run(self, runner):
        """Plot and save loss curves after training."""
        if not self.train_losses and not self.val_losses:
            runner.logger.warning('[LossCurveHook] No loss data to plot')
            return
        
        # Save raw data as JSON
        loss_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        json_path = osp.join(self.out_dir, 'loss_history.json')
        with open(json_path, 'w') as f:
            json.dump(loss_data, f, indent=2)
        runner.logger.info(f'[LossCurveHook] Loss history saved to {json_path}')
        
        # Plot curves
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if self.train_losses:
                epochs, losses = zip(*self.train_losses)
                ax.plot(epochs, losses, 'b-', label='Training Loss', linewidth=2)
            
            if self.val_losses:
                epochs, losses = zip(*self.val_losses)
                ax.plot(epochs, losses, 'r-', label='Validation Loss', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training and Validation Loss Curves', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = osp.join(self.out_dir, self.plot_filename)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            
            runner.logger.info(f'[LossCurveHook] Loss curve saved to {plot_path}')
            
        except Exception as e:
            runner.logger.warning(f'[LossCurveHook] Failed to plot loss curve: {e}')


@HOOKS.register_module()
class MIoULoggerHook(Hook):
    """
    Hook to log mIoU after each validation epoch.
    
    Prints mIoU values in a clear format for monitoring.
    """
    
    def after_val_epoch(self, runner):
        """Print mIoU after validation."""
        if not hasattr(runner, 'eval_results'):
            return
        
        eval_results = runner.eval_results
        if eval_results is None or not isinstance(eval_results, dict):
            return
        
        # Print mIoU if available
        if 'mIoU' in eval_results:
            miou = eval_results['mIoU']
            runner.logger.info(f'[Epoch {runner.epoch + 1}] mIoU: {miou:.4f}')
