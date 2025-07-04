"""
Adaptive Adapters

This module implements various adapter mechanisms that allow the transformer
to adapt to different tasks, domains, and techniques without retraining
the entire model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
from dataclasses import dataclass


@dataclass
class AdapterConfig:
    """Configuration for adapter modules."""
    hidden_size: int = 768
    adapter_size: int = 64
    adapter_dropout: float = 0.1
    adapter_activation: str = "gelu"
    use_bottleneck: bool = True
    use_residual: bool = True


class TaskAdapter(nn.Module):
    """
    Task-specific adapter that adapts the model to different NLP tasks
    like classification, generation, translation, etc.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Task-specific adapters
        self.task_adapters = nn.ModuleDict({
            "classification": self._create_adapter(),
            "generation": self._create_adapter(),
            "translation": self._create_adapter(),
            "summarization": self._create_adapter(),
            "question_answering": self._create_adapter(),
            "named_entity_recognition": self._create_adapter(),
            "sentiment_analysis": self._create_adapter(),
            "text_classification": self._create_adapter(),
        })
        
        # Task embedding
        self.task_embedding = nn.Embedding(len(self.task_adapters), config.hidden_size)
        
        # Task fusion
        self.task_fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
    def _create_adapter(self) -> nn.Module:
        """Create a bottleneck adapter."""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size // 4, self.config.hidden_size),
            nn.Dropout(self.config.dropout)
        )
    
    def forward(self, hidden_states: torch.Tensor, task_type: str) -> torch.Tensor:
        """Apply task-specific adaptation."""
        
        if task_type not in self.task_adapters:
            # Default to generation if task not found
            task_type = "generation"
        
        # Get task adapter
        task_adapter = self.task_adapters[task_type]
        
        # Apply task adapter
        adapted_states = task_adapter(hidden_states)
        
        # Add residual connection
        if self.config.use_residual_connections:
            adapted_states = hidden_states + adapted_states
        
        # Layer normalization
        adapted_states = self.layer_norm(adapted_states)
        
        return adapted_states


class DomainAdapter(nn.Module):
    """
    Domain-specific adapter that adapts the model to different domains
    like medical, legal, technical, etc.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Domain-specific adapters
        self.domain_adapters = nn.ModuleDict({
            "general": self._create_domain_adapter(),
            "medical": self._create_domain_adapter(),
            "legal": self._create_domain_adapter(),
            "technical": self._create_domain_adapter(),
            "scientific": self._create_domain_adapter(),
            "financial": self._create_domain_adapter(),
            "news": self._create_domain_adapter(),
            "academic": self._create_domain_adapter(),
            "conversational": self._create_domain_adapter(),
            "code": self._create_domain_adapter(),
        })
        
        # Domain embedding
        self.domain_embedding = nn.Embedding(len(self.domain_adapters), config.hidden_size)
        
        # Domain-specific vocabulary adaptation
        self.domain_vocab_adaptation = nn.ModuleDict({
            domain: nn.Linear(config.hidden_size, config.hidden_size)
            for domain in self.domain_adapters.keys()
        })
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
    def _create_domain_adapter(self) -> nn.Module:
        """Create a domain-specific adapter."""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size // 2, self.config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size // 4, self.config.hidden_size),
            nn.Dropout(self.config.dropout)
        )
    
    def forward(self, hidden_states: torch.Tensor, domain: str) -> torch.Tensor:
        """Apply domain-specific adaptation."""
        
        if domain not in self.domain_adapters:
            # Default to general domain
            domain = "general"
        
        # Get domain adapter
        domain_adapter = self.domain_adapters[domain]
        
        # Apply domain adapter
        adapted_states = domain_adapter(hidden_states)
        
        # Apply domain-specific vocabulary adaptation
        vocab_adaptation = self.domain_vocab_adaptation[domain]
        adapted_states = vocab_adaptation(adapted_states)
        
        # Add residual connection
        if self.config.use_residual_connections:
            adapted_states = hidden_states + adapted_states
        
        # Layer normalization
        adapted_states = self.layer_norm(adapted_states)
        
        return adapted_states


class TechniqueAdapter(nn.Module):
    """
    Technique-specific adapter that adapts the model to different
    machine learning and deep learning techniques.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Technique-specific adapters
        self.technique_adapters = nn.ModuleDict({
            "standard": self._create_technique_adapter(),
            "few_shot": self._create_technique_adapter(),
            "zero_shot": self._create_technique_adapter(),
            "meta_learning": self._create_technique_adapter(),
            "continual_learning": self._create_technique_adapter(),
            "active_learning": self._create_technique_adapter(),
            "semi_supervised": self._create_technique_adapter(),
            "self_supervised": self._create_technique_adapter(),
            "contrastive_learning": self._create_technique_adapter(),
            "knowledge_distillation": self._create_technique_adapter(),
            "pruning": self._create_technique_adapter(),
            "quantization": self._create_technique_adapter(),
        })
        
        # Technique embedding
        self.technique_embedding = nn.Embedding(len(self.technique_adapters), config.hidden_size)
        
        # Technique-specific optimizations
        self.technique_optimizations = nn.ModuleDict({
            technique: self._create_optimization_module()
            for technique in self.technique_adapters.keys()
        })
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
    def _create_technique_adapter(self) -> nn.Module:
        """Create a technique-specific adapter."""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size // 2, self.config.hidden_size),
            nn.Dropout(self.config.dropout)
        )
    
    def _create_optimization_module(self) -> nn.Module:
        """Create a technique-specific optimization module."""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.config.hidden_size // 4, self.config.hidden_size),
        )
    
    def forward(self, hidden_states: torch.Tensor, technique: str) -> torch.Tensor:
        """Apply technique-specific adaptation."""
        
        if technique not in self.technique_adapters:
            # Default to standard technique
            technique = "standard"
        
        # Get technique adapter
        technique_adapter = self.technique_adapters[technique]
        
        # Apply technique adapter
        adapted_states = technique_adapter(hidden_states)
        
        # Apply technique-specific optimization
        optimization = self.technique_optimizations[technique]
        adapted_states = optimization(adapted_states)
        
        # Add residual connection
        if self.config.use_residual_connections:
            adapted_states = hidden_states + adapted_states
        
        # Layer normalization
        adapted_states = self.layer_norm(adapted_states)
        
        return adapted_states


class MultiTaskAdapter(nn.Module):
    """
    Multi-task adapter that can handle multiple tasks simultaneously
    using task-specific routing.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Task routing network
        self.task_router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Multiple task adapters
        self.task_adapters = nn.ModuleList([
            TaskAdapter(config) for _ in range(4)  # Support up to 4 tasks
        ])
        
        # Task fusion
        self.task_fusion = nn.Linear(config.hidden_size * 4, config.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
    def forward(self, hidden_states: torch.Tensor, task_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply multi-task adaptation."""
        
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Compute task routing weights if not provided
        if task_weights is None:
            task_weights = self.task_router(hidden_states.mean(dim=1))  # [batch_size, 1]
            task_weights = task_weights.expand(-1, 4)  # [batch_size, 4]
            task_weights = F.softmax(task_weights, dim=-1)
        
        # Apply each task adapter
        task_outputs = []
        for i, task_adapter in enumerate(self.task_adapters):
            task_output = task_adapter(hidden_states, f"task_{i}")
            task_outputs.append(task_output)
        
        # Weighted combination of task outputs
        weighted_outputs = []
        for i, task_output in enumerate(task_outputs):
            weight = task_weights[:, i:i+1].unsqueeze(1)  # [batch_size, 1, 1]
            weighted_output = task_output * weight
            weighted_outputs.append(weighted_output)
        
        # Concatenate and fuse
        combined_output = torch.cat(weighted_outputs, dim=-1)
        fused_output = self.task_fusion(combined_output)
        
        # Add residual connection
        if self.config.use_residual_connections:
            fused_output = hidden_states + fused_output
        
        # Layer normalization
        fused_output = self.layer_norm(fused_output)
        
        return fused_output


class DynamicAdapter(nn.Module):
    """
    Dynamic adapter that automatically learns which adaptation to apply
    based on the input characteristics.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input analysis network
        self.input_analyzer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 128),  # Feature vector
        )
        
        # Adaptation selector
        self.adaptation_selector = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 16),  # Number of adaptation strategies
            nn.Softmax(dim=-1)
        )
        
        # Adaptation strategies
        self.adaptation_strategies = nn.ModuleList([
            self._create_adaptation_strategy() for _ in range(16)
        ])
        
        # Output fusion
        self.output_fusion = nn.Linear(config.hidden_size * 16, config.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
    def _create_adaptation_strategy(self) -> nn.Module:
        """Create an adaptation strategy."""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size // 2, self.config.hidden_size),
            nn.Dropout(self.config.dropout)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply dynamic adaptation."""
        
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Analyze input characteristics
        input_features = self.input_analyzer(hidden_states.mean(dim=1))  # [batch_size, 128]
        
        # Select adaptation strategies
        adaptation_weights = self.adaptation_selector(input_features)  # [batch_size, 16]
        
        # Apply each adaptation strategy
        strategy_outputs = []
        for i, strategy in enumerate(self.adaptation_strategies):
            strategy_output = strategy(hidden_states)
            weight = adaptation_weights[:, i:i+1].unsqueeze(1)  # [batch_size, 1, 1]
            weighted_output = strategy_output * weight
            strategy_outputs.append(weighted_output)
        
        # Concatenate and fuse
        combined_output = torch.cat(strategy_outputs, dim=-1)
        fused_output = self.output_fusion(combined_output)
        
        # Add residual connection
        if self.config.use_residual_connections:
            fused_output = hidden_states + fused_output
        
        # Layer normalization
        fused_output = self.layer_norm(fused_output)
        
        return fused_output


class CrossModalAdapter(nn.Module):
    """
    Cross-modal adapter that adapts the model to handle different modalities
    like text, image, audio, etc.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Modality-specific adapters
        self.modality_adapters = nn.ModuleDict({
            "text": self._create_modality_adapter(),
            "image": self._create_modality_adapter(),
            "audio": self._create_modality_adapter(),
            "video": self._create_modality_adapter(),
            "multimodal": self._create_modality_adapter(),
        })
        
        # Cross-modal fusion
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        
        # Modality embedding
        self.modality_embedding = nn.Embedding(len(self.modality_adapters), config.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
    def _create_modality_adapter(self) -> nn.Module:
        """Create a modality-specific adapter."""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Dropout(self.config.dropout)
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        modality: str,
        other_modality_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply cross-modal adaptation."""
        
        if modality not in self.modality_adapters:
            modality = "text"
        
        # Apply modality-specific adapter
        modality_adapter = self.modality_adapters[modality]
        adapted_states = modality_adapter(hidden_states)
        
        # Cross-modal fusion if other modality is provided
        if other_modality_states is not None:
            # Ensure same sequence length
            if other_modality_states.size(1) != adapted_states.size(1):
                other_modality_states = F.interpolate(
                    other_modality_states.transpose(1, 2),
                    size=adapted_states.size(1),
                    mode='linear'
                ).transpose(1, 2)
            
            # Concatenate modalities
            combined_states = torch.cat([adapted_states, other_modality_states], dim=-1)
            fused_states = self.cross_modal_fusion(combined_states)
        else:
            fused_states = adapted_states
        
        # Add residual connection
        if self.config.use_residual_connections:
            fused_states = hidden_states + fused_states
        
        # Layer normalization
        fused_states = self.layer_norm(fused_states)
        
        return fused_states 