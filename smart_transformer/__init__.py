"""
Smart Transformer - An Adaptive and Intelligent Transformer Architecture

This package implements a state-of-the-art transformer that adapts to various
ML and deep learning techniques to outperform existing transformers.
"""

from .core import SmartTransformer, AdaptiveConfig
from .attention import AdaptiveAttention, MultiScaleAttention
from .adapters import TaskAdapter, DomainAdapter, TechniqueAdapter
from .optimization import AdaptiveOptimizer, DynamicLearningRate
from .training import SmartTrainer, AdaptiveTrainingLoop
from .evaluation import SmartEvaluator, PerformanceAnalyzer

__version__ = "1.0.0"
__author__ = "Smart Transformer Team"

__all__ = [
    "SmartTransformer",
    "AdaptiveConfig", 
    "AdaptiveAttention",
    "MultiScaleAttention",
    "TaskAdapter",
    "DomainAdapter", 
    "TechniqueAdapter",
    "AdaptiveOptimizer",
    "DynamicLearningRate",
    "SmartTrainer",
    "AdaptiveTrainingLoop",
    "SmartEvaluator",
    "PerformanceAnalyzer"
] 