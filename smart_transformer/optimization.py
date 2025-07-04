"""
Adaptive Optimization

This module implements advanced optimization techniques that adapt to
the training dynamics and model characteristics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple, Union, Callable
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_clip_val: float = 1.0
    use_adaptive_optimization: bool = True
    use_dynamic_lr: bool = True
    use_gradient_accumulation: bool = True
    accumulation_steps: int = 4


class AdaptiveOptimizer:
    """
    Adaptive optimizer that automatically selects the best optimization
    strategy based on training dynamics.
    """
    
    def __init__(self, model: nn.Module, config: OptimizationConfig):
        self.config = config
        self.model = model
        
        # Initialize multiple optimizers
        self.optimizers = {
            'adamw': optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            ),
            'adam': optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            ),
            'sgd': optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            ),
            'lion': self._create_lion_optimizer(),
        }
        
        # Current optimizer
        self.current_optimizer = 'adamw'
        self.optimizer = self.optimizers[self.current_optimizer]
        
        # Learning rate schedulers
        self.schedulers = {
            'cosine': CosineAnnealingLR(self.optimizer, T_max=config.max_steps),
            'plateau': ReduceLROnPlateau(self.optimizer, mode='min', patience=5),
            'warmup_cosine': self._create_warmup_cosine_scheduler(),
            'linear': self._create_linear_scheduler(),
        }
        
        # Current scheduler
        self.current_scheduler = 'warmup_cosine'
        self.scheduler = self.schedulers[self.current_scheduler]
        
        # Performance tracking
        self.loss_history = []
        self.gradient_norms = []
        self.learning_rates = []
        
        # Adaptation parameters
        self.adaptation_threshold = 0.1
        self.adaptation_frequency = 1000
        
    def _create_lion_optimizer(self):
        """Create Lion optimizer (if available)."""
        try:
            from lion_pytorch import Lion
            return Lion(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        except ImportError:
            # Fallback to AdamW if Lion not available
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _create_warmup_cosine_scheduler(self):
        """Create warmup cosine scheduler."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            progress = float(step - self.config.warmup_steps) / float(
                max(1, self.config.max_steps - self.config.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _create_linear_scheduler(self):
        """Create linear scheduler."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            progress = float(step - self.config.warmup_steps) / float(
                max(1, self.config.max_steps - self.config.warmup_steps)
            )
            return max(0.0, 1.0 - progress)
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def step(self, loss: float, step: int):
        """Perform optimization step with adaptation."""
        
        # Track performance
        self.loss_history.append(loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        # Compute gradient norm
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        # Gradient clipping
        if self.config.use_adaptive_optimization:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip_val
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()
        
        # Adaptive optimization selection
        if step % self.adaptation_frequency == 0 and self.config.use_adaptive_optimization:
            self._adapt_optimization_strategy(step)
    
    def _adapt_optimization_strategy(self, step: int):
        """Adapt optimization strategy based on training dynamics."""
        
        if len(self.loss_history) < 100:
            return
        
        # Analyze recent performance
        recent_losses = self.loss_history[-100:]
        recent_gradients = self.gradient_norms[-100:]
        
        # Compute metrics
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        gradient_mean = np.mean(recent_gradients)
        gradient_std = np.std(recent_gradients)
        
        # Adapt optimizer
        if loss_trend > self.adaptation_threshold:
            # Loss is increasing, try different optimizer
            if self.current_optimizer == 'adamw':
                self._switch_optimizer('lion')
            elif self.current_optimizer == 'lion':
                self._switch_optimizer('adam')
            else:
                self._switch_optimizer('sgd')
        
        # Adapt scheduler
        if gradient_std > gradient_mean * 0.5:
            # High gradient variance, use more stable scheduler
            if self.current_scheduler != 'plateau':
                self._switch_scheduler('plateau')
        else:
            # Stable gradients, use more aggressive scheduler
            if self.current_scheduler != 'warmup_cosine':
                self._switch_scheduler('warmup_cosine')
    
    def _switch_optimizer(self, optimizer_name: str):
        """Switch to a different optimizer."""
        if optimizer_name == self.current_optimizer:
            return
        
        # Save current state
        old_state = self.optimizer.state_dict()
        
        # Switch optimizer
        self.current_optimizer = optimizer_name
        self.optimizer = self.optimizers[optimizer_name]
        
        # Transfer state if possible
        try:
            self.optimizer.load_state_dict(old_state)
        except:
            # If state transfer fails, continue with new optimizer
            pass
        
        # Update schedulers
        for scheduler in self.schedulers.values():
            if hasattr(scheduler, 'optimizer'):
                scheduler.optimizer = self.optimizer
    
    def _switch_scheduler(self, scheduler_name: str):
        """Switch to a different scheduler."""
        if scheduler_name == self.current_scheduler:
            return
        
        self.current_scheduler = scheduler_name
        self.scheduler = self.schedulers[scheduler_name]


class DynamicLearningRate:
    """
    Dynamic learning rate scheduler that adapts based on training dynamics.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.base_lr = config.learning_rate
        self.current_lr = config.learning_rate
        
        # Learning rate bounds
        self.min_lr = config.learning_rate * 0.01
        self.max_lr = config.learning_rate * 10.0
        
        # Performance tracking
        self.loss_history = []
        self.lr_history = []
        
        # Adaptation parameters
        self.patience = 5
        self.factor = 0.5
        self.cooldown = 0
        self.best_loss = float('inf')
        self.bad_epochs = 0
        
    def step(self, loss: float, epoch: int) -> float:
        """Update learning rate based on loss."""
        
        self.loss_history.append(loss)
        self.lr_history.append(self.current_lr)
        
        # Check if loss improved
        if loss < self.best_loss:
            self.best_loss = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        
        # Reduce learning rate if no improvement
        if self.bad_epochs >= self.patience:
            self.current_lr = max(self.min_lr, self.current_lr * self.factor)
            self.bad_epochs = 0
        
        # Adaptive learning rate based on loss trend
        if len(self.loss_history) >= 10:
            recent_trend = self._compute_loss_trend()
            self._adapt_lr_based_on_trend(recent_trend)
        
        return self.current_lr
    
    def _compute_loss_trend(self) -> float:
        """Compute the trend of recent losses."""
        recent_losses = self.loss_history[-10:]
        x = np.arange(len(recent_losses))
        slope = np.polyfit(x, recent_losses, 1)[0]
        return slope
    
    def _adapt_lr_based_on_trend(self, trend: float):
        """Adapt learning rate based on loss trend."""
        if trend > 0.01:  # Loss increasing
            self.current_lr = max(self.min_lr, self.current_lr * 0.9)
        elif trend < -0.01:  # Loss decreasing
            self.current_lr = min(self.max_lr, self.current_lr * 1.1)
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


class GradientAccumulator:
    """
    Gradient accumulator for effective large batch training.
    """
    
    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.gradients = []
    
    def accumulate(self, loss: torch.Tensor, model: nn.Module):
        """Accumulate gradients."""
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.current_step += 1
        
        # Store gradients if not at accumulation boundary
        if self.current_step % self.accumulation_steps != 0:
            self.gradients.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
    
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return self.current_step % self.accumulation_steps == 0
    
    def reset(self):
        """Reset accumulator."""
        self.current_step = 0
        self.gradients = []


class AdaptiveWeightDecay:
    """
    Adaptive weight decay that adjusts based on parameter importance.
    """
    
    def __init__(self, model: nn.Module, base_weight_decay: float = 0.01):
        self.base_weight_decay = base_weight_decay
        self.model = model
        
        # Parameter importance tracking
        self.parameter_importance = {}
        for name, param in model.named_parameters():
            self.parameter_importance[name] = 1.0
    
    def update_importance(self, gradients: Dict[str, torch.Tensor]):
        """Update parameter importance based on gradients."""
        for name, grad in gradients.items():
            if grad is not None:
                # Compute importance as gradient magnitude
                importance = grad.norm().item()
                self.parameter_importance[name] = importance
    
    def get_weight_decay(self, param_name: str) -> float:
        """Get adaptive weight decay for parameter."""
        importance = self.parameter_importance.get(param_name, 1.0)
        
        # Higher importance -> lower weight decay
        adaptive_decay = self.base_weight_decay / (1.0 + importance)
        
        return adaptive_decay


class LearningRateFinder:
    """
    Learning rate finder that automatically discovers optimal learning rates.
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.original_lr = optimizer.param_groups[0]['lr']
        
        # Tracking
        self.lr_history = []
        self.loss_history = []
        
        # Search parameters
        self.start_lr = 1e-7
        self.end_lr = 1.0
        self.num_iterations = 100
        
    def find_lr(self, train_loader, loss_fn):
        """Find optimal learning rate."""
        
        # Reset optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.start_lr
        
        # Exponential learning rate schedule
        gamma = (self.end_lr / self.start_lr) ** (1 / self.num_iterations)
        
        iteration = 0
        for batch in train_loader:
            if iteration >= self.num_iterations:
                break
            
            # Forward pass
            outputs = self.model(batch)
            loss = loss_fn(outputs, batch['labels'])
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Track
            self.lr_history.append(self.optimizer.param_groups[0]['lr'])
            self.loss_history.append(loss.item())
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= gamma
            
            iteration += 1
        
        # Find optimal learning rate
        optimal_lr = self._find_optimal_lr()
        
        # Reset to original learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.original_lr
        
        return optimal_lr
    
    def _find_optimal_lr(self) -> float:
        """Find optimal learning rate from loss curve."""
        # Find the learning rate where loss starts to increase rapidly
        losses = np.array(self.loss_history)
        lrs = np.array(self.lr_history)
        
        # Compute loss gradient
        loss_grad = np.gradient(losses)
        
        # Find point where gradient becomes positive and large
        threshold = np.std(loss_grad) * 2
        optimal_idx = np.where(loss_grad > threshold)[0]
        
        if len(optimal_idx) > 0:
            optimal_idx = optimal_idx[0]
            return lrs[optimal_idx]
        else:
            # Fallback to minimum loss
            return lrs[np.argmin(losses)]


class AdaptiveGradientClipping:
    """
    Adaptive gradient clipping that adjusts clipping threshold based on
    gradient statistics.
    """
    
    def __init__(self, initial_threshold: float = 1.0):
        self.threshold = initial_threshold
        self.gradient_norms = []
        
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients adaptively."""
        
        # Compute gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        self.gradient_norms.append(total_norm)
        
        # Adapt threshold based on gradient statistics
        if len(self.gradient_norms) >= 100:
            mean_norm = np.mean(self.gradient_norms[-100:])
            std_norm = np.std(self.gradient_norms[-100:])
            
            # Adjust threshold based on gradient distribution
            if total_norm > mean_norm + 2 * std_norm:
                self.threshold = max(0.1, self.threshold * 0.9)
            elif total_norm < mean_norm - std_norm:
                self.threshold = min(10.0, self.threshold * 1.1)
        
        # Apply clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.threshold)
        
        return total_norm 