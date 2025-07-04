"""
Smart Training

This module implements advanced training strategies that adapt to the
training dynamics and model performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import time
import logging
from tqdm import tqdm
import wandb
from dataclasses import dataclass

from .optimization import AdaptiveOptimizer, OptimizationConfig, DynamicLearningRate
from .evaluation import SmartEvaluator


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 16
    num_epochs: int = 10
    max_steps: int = 100000
    eval_steps: int = 1000
    save_steps: int = 5000
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    use_adaptive_training: bool = True
    use_early_stopping: bool = True
    patience: int = 5
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_wandb: bool = True


class SmartTrainer:
    """
    Smart trainer that adapts training strategies based on model performance
    and training dynamics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: TrainingConfig = None,
        loss_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        # Initialize components
        self.optimizer_config = OptimizationConfig(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            gradient_clip_val=self.config.max_grad_norm,
        )
        
        self.optimizer = AdaptiveOptimizer(model, self.optimizer_config)
        self.evaluator = SmartEvaluator()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gradient_norm': [],
        }
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Setup wandb
        if self.config.use_wandb:
            wandb.init(project="smart-transformer", config=vars(self.config))
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('smart_transformer_training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def train(self):
        """Main training loop."""
        
        self.logger.info("Starting smart training...")
        
        # Setup data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self._train_epoch(train_loader)
            
            # Validate
            if self.val_dataset:
                val_loss = self._validate_epoch(val_loader)
                
                # Early stopping check
                if self.config.use_early_stopping:
                    if self._should_stop_early(val_loss):
                        self.logger.info("Early stopping triggered")
                        break
            
            # Log progress
            self._log_epoch_progress(train_loss, val_loss if self.val_dataset else None)
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint()
        
        self.logger.info("Training completed!")
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            loss = self._training_step(batch)
            
            # Update progress
            total_loss += loss
            avg_loss = total_loss / (batch_idx + 1)
            
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Evaluation
            if self.current_step % self.config.eval_steps == 0 and self.val_dataset:
                self._evaluate_and_adapt()
            
            # Save checkpoint
            if self.current_step % self.config.save_steps == 0:
                self._save_checkpoint()
            
            self.current_step += 1
        
        return total_loss / num_batches
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        
        # Move to device
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Compute loss
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Handle different label formats
        if 'labels' in batch:
            labels = batch['labels']
        elif 'input_ids' in batch:
            # For language modeling, shift labels
            labels = batch['input_ids'][:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous()
        
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (self.current_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Optimizer step
            self.optimizer.step(loss.item(), self.current_step)
            self.model.zero_grad()
        
        return loss.item()
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Compute loss
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Handle different label formats
                if 'labels' in batch:
                    labels = batch['labels']
                elif 'input_ids' in batch:
                    labels = batch['input_ids'][:, 1:].contiguous()
                    logits = logits[:, :-1, :].contiguous()
                
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _evaluate_and_adapt(self):
        """Evaluate model and adapt training strategy."""
        
        # Quick validation
        val_loss = self._quick_validation()
        
        # Adapt training strategy
        if self.config.use_adaptive_training:
            self._adapt_training_strategy(val_loss)
        
        # Update best loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
    
    def _quick_validation(self) -> float:
        """Perform quick validation on a subset of validation data."""
        
        if not self.val_dataset:
            return float('inf')
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        self.model.eval()
        total_loss = 0
        num_batches = min(10, len(val_loader))  # Quick validation
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_batches:
                    break
                
                # Move to device
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Compute loss
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                if 'labels' in batch:
                    labels = batch['labels']
                elif 'input_ids' in batch:
                    labels = batch['input_ids'][:, 1:].contiguous()
                    logits = logits[:, :-1, :].contiguous()
                
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _adapt_training_strategy(self, val_loss: float):
        """Adapt training strategy based on validation loss."""
        
        # Analyze training dynamics
        recent_losses = self.training_history['train_loss'][-100:]
        recent_val_losses = self.training_history['val_loss'][-100:]
        
        if len(recent_losses) < 50:
            return
        
        # Compute trends
        train_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        val_trend = np.polyfit(range(len(recent_val_losses)), recent_val_losses, 1)[0]
        
        # Adapt learning rate
        if val_trend > 0.01:  # Validation loss increasing
            self._reduce_learning_rate()
        elif val_trend < -0.01 and train_trend < -0.01:  # Both decreasing
            self._increase_learning_rate()
        
        # Adapt batch size if needed
        if val_loss > self.best_val_loss * 1.1:
            self._adapt_batch_size()
    
    def _reduce_learning_rate(self):
        """Reduce learning rate."""
        for param_group in self.optimizer.optimizer.param_groups:
            param_group['lr'] *= 0.9
        self.logger.info("Reduced learning rate")
    
    def _increase_learning_rate(self):
        """Increase learning rate."""
        for param_group in self.optimizer.optimizer.param_groups:
            param_group['lr'] *= 1.1
        self.logger.info("Increased learning rate")
    
    def _adapt_batch_size(self):
        """Adapt batch size based on performance."""
        # This is a simplified implementation
        # In practice, you'd need to handle memory constraints
        pass
    
    def _should_stop_early(self, val_loss: float) -> bool:
        """Check if training should stop early."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.patience
    
    def _log_epoch_progress(self, train_loss: float, val_loss: Optional[float]):
        """Log training progress."""
        
        # Update history
        self.training_history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.training_history['val_loss'].append(val_loss)
        
        current_lr = self.optimizer.optimizer.param_groups[0]['lr']
        self.training_history['learning_rate'].append(current_lr)
        
        # Log to console
        log_msg = f"Epoch {self.current_epoch + 1}: Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            log_msg += f", Val Loss: {val_loss:.4f}"
        log_msg += f", LR: {current_lr:.2e}"
        
        self.logger.info(log_msg)
        
        # Log to wandb
        if self.config.use_wandb:
            log_dict = {
                'epoch': self.current_epoch + 1,
                'train_loss': train_loss,
                'learning_rate': current_lr,
            }
            if val_loss is not None:
                log_dict['val_loss'] = val_loss
            
            wandb.log(log_dict)
    
    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config,
        }
        
        torch.save(checkpoint, f'checkpoint_epoch_{self.current_epoch}_step_{self.current_step}.pt')
        self.logger.info(f"Saved checkpoint at epoch {self.current_epoch}, step {self.current_step}")


class AdaptiveTrainingLoop:
    """
    Adaptive training loop that automatically adjusts training parameters
    based on model performance and training dynamics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: TrainingConfig = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        
        # Initialize components
        self.optimizer = AdaptiveOptimizer(model, OptimizationConfig())
        self.evaluator = SmartEvaluator()
        
        # Training state
        self.current_step = 0
        self.best_metric = float('inf')
        self.training_stats = {
            'loss': [],
            'gradient_norm': [],
            'learning_rate': [],
            'batch_size': [],
        }
        
    def run(self, max_steps: int = 100000):
        """Run adaptive training loop."""
        
        print("Starting adaptive training loop...")
        
        while self.current_step < max_steps:
            # Training step
            loss = self._adaptive_training_step()
            
            # Evaluation
            if self.current_step % self.config.eval_steps == 0:
                self._evaluate_and_adapt()
            
            # Logging
            if self.current_step % 100 == 0:
                self._log_progress()
            
            self.current_step += 1
    
    def _adaptive_training_step(self) -> float:
        """Single adaptive training step."""
        
        # Get batch
        try:
            batch = next(self.train_iter)
        except (StopIteration, AttributeError):
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
        
        # Forward pass
        outputs = self.model(**batch)
        loss = self._compute_loss(outputs, batch)
        
        # Backward pass
        loss.backward()
        
        # Adaptive gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step(loss.item(), self.current_step)
        self.model.zero_grad()
        
        # Update statistics
        self.training_stats['loss'].append(loss.item())
        self.training_stats['gradient_norm'].append(grad_norm)
        self.training_stats['learning_rate'].append(
            self.optimizer.optimizer.param_groups[0]['lr']
        )
        
        return loss.item()
    
    def _compute_loss(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss based on outputs and batch."""
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Handle different label formats
        if 'labels' in batch:
            labels = batch['labels']
        elif 'input_ids' in batch:
            labels = batch['input_ids'][:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous()
        
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    def _evaluate_and_adapt(self):
        """Evaluate model and adapt training strategy."""
        
        if not self.val_loader:
            return
        
        # Quick evaluation
        val_loss = self._quick_evaluation()
        
        # Adapt training strategy
        self._adapt_strategy(val_loss)
    
    def _quick_evaluation(self) -> float:
        """Perform quick evaluation."""
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if num_batches >= 10:  # Quick evaluation
                    break
                
                outputs = self.model(**batch)
                loss = self._compute_loss(outputs, batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def _adapt_strategy(self, val_loss: float):
        """Adapt training strategy based on validation loss."""
        
        # Analyze recent performance
        recent_losses = self.training_stats['loss'][-100:]
        recent_gradients = self.training_stats['gradient_norm'][-100:]
        
        if len(recent_losses) < 50:
            return
        
        # Compute metrics
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        gradient_mean = np.mean(recent_gradients)
        
        # Adapt based on trends
        if loss_trend > 0.01:  # Loss increasing
            self._adapt_learning_rate(reduce=True)
        elif loss_trend < -0.01:  # Loss decreasing
            self._adapt_learning_rate(reduce=False)
        
        # Adapt based on gradient statistics
        if gradient_mean > 1.0:
            self._adapt_gradient_clipping()
    
    def _adapt_learning_rate(self, reduce: bool = True):
        """Adapt learning rate."""
        factor = 0.9 if reduce else 1.1
        for param_group in self.optimizer.optimizer.param_groups:
            param_group['lr'] *= factor
    
    def _adapt_gradient_clipping(self):
        """Adapt gradient clipping threshold."""
        # This is a simplified implementation
        pass
    
    def _log_progress(self):
        """Log training progress."""
        
        recent_loss = np.mean(self.training_stats['loss'][-100:])
        current_lr = self.optimizer.optimizer.param_groups[0]['lr']
        
        print(f"Step {self.current_step}: Loss: {recent_loss:.4f}, LR: {current_lr:.2e}")


class CurriculumTrainer:
    """
    Curriculum trainer that gradually increases task difficulty during training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        config: TrainingConfig = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config or TrainingConfig()
        
        # Curriculum parameters
        self.curriculum_stages = [
            {'max_length': 64, 'epochs': 2},
            {'max_length': 128, 'epochs': 3},
            {'max_length': 256, 'epochs': 3},
            {'max_length': 512, 'epochs': 2},
        ]
        
        self.current_stage = 0
        self.stage_epoch = 0
        
    def train(self):
        """Train with curriculum learning."""
        
        for stage_idx, stage in enumerate(self.curriculum_stages):
            self.current_stage = stage_idx
            self.stage_epoch = 0
            
            print(f"Starting curriculum stage {stage_idx + 1}: max_length={stage['max_length']}")
            
            # Create stage-specific dataset
            stage_dataset = self._create_stage_dataset(stage['max_length'])
            stage_loader = DataLoader(
                stage_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4
            )
            
            # Train for this stage
            for epoch in range(stage['epochs']):
                self.stage_epoch = epoch
                self._train_stage_epoch(stage_loader)
    
    def _create_stage_dataset(self, max_length: int) -> Dataset:
        """Create dataset for current curriculum stage."""
        # This is a simplified implementation
        # In practice, you'd filter the dataset based on sequence length
        return self.train_dataset
    
    def _train_stage_epoch(self, stage_loader: DataLoader):
        """Train for one epoch in current curriculum stage."""
        
        self.model.train()
        total_loss = 0
        num_batches = len(stage_loader)
        
        for batch_idx, batch in enumerate(tqdm(stage_loader, desc=f"Stage {self.current_stage + 1}, Epoch {self.stage_epoch + 1}")):
            # Training step
            loss = self._training_step(batch)
            total_loss += loss
        
        avg_loss = total_loss / num_batches
        print(f"Stage {self.current_stage + 1}, Epoch {self.stage_epoch + 1}: Loss: {avg_loss:.4f}")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Compute loss
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        if 'labels' in batch:
            labels = batch['labels']
        elif 'input_ids' in batch:
            labels = batch['input_ids'][:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous()
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Optimizer step (simplified)
        for param in self.model.parameters():
            if param.grad is not None:
                param.data -= 1e-4 * param.grad
                param.grad.zero_()
        
        return loss.item() 