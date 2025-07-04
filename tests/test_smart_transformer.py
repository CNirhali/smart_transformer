"""
Tests for Smart Transformer

This module contains comprehensive tests for all components
of the Smart Transformer implementation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from smart_transformer import SmartTransformer, AdaptiveConfig
from smart_transformer.attention import AdaptiveAttention, MultiScaleAttention
from smart_transformer.adapters import TaskAdapter, DomainAdapter, TechniqueAdapter
from smart_transformer.optimization import AdaptiveOptimizer, OptimizationConfig
from smart_transformer.training import SmartTrainer, TrainingConfig
from smart_transformer.evaluation import SmartEvaluator, EvaluationConfig


class TestAdaptiveConfig:
    """Test AdaptiveConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = AdaptiveConfig()
        
        assert config.vocab_size == 50257
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_attention_heads == 12
        assert config.use_adaptive_attention is True
        assert config.use_task_adapters is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AdaptiveConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=6,
            num_attention_heads=8,
            use_adaptive_attention=False,
            use_task_adapters=False,
        )
        
        assert config.vocab_size == 1000
        assert config.hidden_size == 256
        assert config.num_layers == 6
        assert config.num_attention_heads == 8
        assert config.use_adaptive_attention is False
        assert config.use_task_adapters is False
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid hidden_size
        with pytest.raises(AssertionError):
            AdaptiveConfig(hidden_size=100, num_attention_heads=12)
        
        # Test invalid intermediate_size
        with pytest.raises(AssertionError):
            AdaptiveConfig(intermediate_size=100, hidden_size=768)


class TestSmartTransformer:
    """Test SmartTransformer class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return AdaptiveConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_attention_heads=8,
            intermediate_size=512,
            max_position_embeddings=128,
            use_adaptive_attention=True,
            use_task_adapters=True,
            use_domain_adapters=True,
            use_technique_adapters=True,
        )
    
    @pytest.fixture
    def model(self, config):
        """Create a test model."""
        return SmartTransformer(config)
    
    def test_model_initialization(self, model, config):
        """Test model initialization."""
        assert isinstance(model, SmartTransformer)
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        
        # Check that embeddings are tied
        assert model.lm_head.weight is model.embeddings.token_embeddings.weight
    
    def test_forward_pass(self, model):
        """Test forward pass."""
        batch_size = 2
        seq_length = 16
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Test basic forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_length, 1000)
    
    def test_forward_pass_with_adapters(self, model):
        """Test forward pass with adapters."""
        batch_size = 2
        seq_length = 16
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Test with task adapter
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_type="classification",
            domain="medical",
            technique="few_shot"
        )
        
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
    
    def test_generate(self, model):
        """Test text generation."""
        prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        
        generated = model.generate(
            input_ids=prompt,
            max_length=10,
            do_sample=False,
            pad_token_id=0,
            eos_token_id=2
        )
        
        assert generated.shape[0] == 1
        assert generated.shape[1] >= prompt.shape[1]
        assert generated.shape[1] <= 10
    
    def test_model_save_load(self, model, tmp_path):
        """Test model saving and loading."""
        # Save model
        save_path = tmp_path / "test_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model.config,
        }, save_path)
        
        # Load model
        checkpoint = torch.load(save_path)
        new_model = SmartTransformer(checkpoint['config'])
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test that models are equivalent
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        
        with torch.no_grad():
            outputs1 = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs2 = new_model(input_ids=input_ids, attention_mask=attention_mask)
        
        torch.testing.assert_close(outputs1['logits'], outputs2['logits'])


class TestAdaptiveAttention:
    """Test AdaptiveAttention class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return AdaptiveConfig(
            hidden_size=256,
            num_attention_heads=8,
            use_flash_attention=False,  # Disable for testing
        )
    
    @pytest.fixture
    def attention(self, config):
        """Create a test attention module."""
        return AdaptiveAttention(config)
    
    def test_attention_initialization(self, attention):
        """Test attention initialization."""
        assert isinstance(attention, AdaptiveAttention)
        assert attention.num_heads == 8
        assert attention.head_dim == 32
    
    def test_attention_forward(self, attention):
        """Test attention forward pass."""
        batch_size = 2
        seq_length = 16
        hidden_size = 256
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length)
        
        outputs = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        
        assert len(outputs) >= 1
        assert outputs[0].shape == (batch_size, seq_length, hidden_size)


class TestAdapters:
    """Test adapter classes."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return AdaptiveConfig(
            hidden_size=256,
            dropout=0.1,
        )
    
    def test_task_adapter(self, config):
        """Test TaskAdapter."""
        adapter = TaskAdapter(config)
        
        batch_size = 2
        seq_length = 16
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
        
        # Test with different task types
        task_types = ["classification", "generation", "translation"]
        
        for task_type in task_types:
            adapted_states = adapter(hidden_states, task_type)
            assert adapted_states.shape == hidden_states.shape
    
    def test_domain_adapter(self, config):
        """Test DomainAdapter."""
        adapter = DomainAdapter(config)
        
        batch_size = 2
        seq_length = 16
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
        
        # Test with different domains
        domains = ["general", "medical", "legal", "technical"]
        
        for domain in domains:
            adapted_states = adapter(hidden_states, domain)
            assert adapted_states.shape == hidden_states.shape
    
    def test_technique_adapter(self, config):
        """Test TechniqueAdapter."""
        adapter = TechniqueAdapter(config)
        
        batch_size = 2
        seq_length = 16
        hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
        
        # Test with different techniques
        techniques = ["standard", "few_shot", "zero_shot", "meta_learning"]
        
        for technique in techniques:
            adapted_states = adapter(hidden_states, technique)
            assert adapted_states.shape == hidden_states.shape


class TestOptimization:
    """Test optimization classes."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return OptimizationConfig(
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=1000,
        )
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        config = AdaptiveConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
        )
        return SmartTransformer(config)
    
    def test_adaptive_optimizer(self, model, config):
        """Test AdaptiveOptimizer."""
        optimizer = AdaptiveOptimizer(model, config)
        
        assert isinstance(optimizer, AdaptiveOptimizer)
        assert optimizer.current_optimizer == 'adamw'
        assert optimizer.current_scheduler == 'warmup_cosine'
    
    def test_optimizer_step(self, model, config):
        """Test optimizer step."""
        optimizer = AdaptiveOptimizer(model, config)
        
        # Create dummy loss
        loss = torch.tensor(1.0, requires_grad=True)
        
        # Test step
        optimizer.step(loss.item(), 0)
        
        # Check that learning rate history is updated
        assert len(optimizer.learning_rates) > 0
        assert len(optimizer.loss_history) > 0


class TestTraining:
    """Test training classes."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return TrainingConfig(
            batch_size=4,
            num_epochs=2,
            learning_rate=1e-4,
            use_adaptive_training=True,
        )
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        model_config = AdaptiveConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
        )
        return SmartTransformer(model_config)
    
    @pytest.fixture
    def dataset(self):
        """Create a test dataset."""
        input_ids = torch.randint(0, 100, (20, 16))
        attention_mask = torch.ones(20, 16)
        labels = input_ids.clone()
        
        from torch.utils.data import TensorDataset
        return TensorDataset(input_ids, attention_mask, labels)
    
    def test_smart_trainer_initialization(self, model, dataset, config):
        """Test SmartTrainer initialization."""
        trainer = SmartTrainer(
            model=model,
            train_dataset=dataset,
            config=config
        )
        
        assert isinstance(trainer, SmartTrainer)
        assert trainer.config == config
        assert trainer.model == model
    
    def test_training_step(self, model, dataset, config):
        """Test training step."""
        trainer = SmartTrainer(
            model=model,
            train_dataset=dataset,
            config=config
        )
        
        # Create a batch
        batch = (torch.randint(0, 100, (2, 8)), 
                torch.ones(2, 8), 
                torch.randint(0, 100, (2, 8)))
        
        # Test training step
        loss = trainer._training_step({
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        })
        
        assert isinstance(loss, float)
        assert loss > 0


class TestEvaluation:
    """Test evaluation classes."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return EvaluationConfig(
            batch_size=4,
            compute_perplexity=True,
            compute_accuracy=True,
        )
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        model_config = AdaptiveConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
        )
        return SmartTransformer(model_config)
    
    @pytest.fixture
    def dataset(self):
        """Create a test dataset."""
        input_ids = torch.randint(0, 100, (10, 16))
        attention_mask = torch.ones(10, 16)
        labels = input_ids.clone()
        
        from torch.utils.data import TensorDataset
        return TensorDataset(input_ids, attention_mask, labels)
    
    def test_evaluator_initialization(self, config):
        """Test SmartEvaluator initialization."""
        evaluator = SmartEvaluator(config)
        
        assert isinstance(evaluator, SmartEvaluator)
        assert evaluator.config == config
    
    def test_evaluation(self, model, dataset, config):
        """Test model evaluation."""
        evaluator = SmartEvaluator(config)
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Test evaluation
        results = evaluator.evaluate(
            model=model,
            test_loader=dataloader,
            task_type="language_modeling",
            save_results=False
        )
        
        assert isinstance(results, dict)
        assert 'loss' in results
    
    def test_performance_analyzer(self, model, dataset, config):
        """Test PerformanceAnalyzer."""
        from smart_transformer.evaluation import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Test performance analysis
        analysis = analyzer.analyze_model_performance(model, dataloader)
        
        assert isinstance(analysis, dict)
        assert 'computational_complexity' in analysis
        assert 'memory_usage' in analysis
        assert 'latency_analysis' in analysis
        assert 'throughput_analysis' in analysis


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training workflow."""
        # Create configuration
        config = AdaptiveConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            use_adaptive_attention=True,
            use_task_adapters=True,
        )
        
        # Create model
        model = SmartTransformer(config)
        
        # Create dataset
        input_ids = torch.randint(0, 100, (20, 16))
        attention_mask = torch.ones(20, 16)
        labels = input_ids.clone()
        
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Create trainer
        training_config = TrainingConfig(
            batch_size=4,
            num_epochs=1,
            learning_rate=1e-4,
        )
        
        trainer = SmartTrainer(
            model=model,
            train_dataset=dataset,
            config=training_config
        )
        
        # Test that everything works together
        assert model is not None
        assert trainer is not None
        
        # Test forward pass
        batch = next(iter(dataloader))
        outputs = model(
            input_ids=batch[0],
            attention_mask=batch[1],
            task_type="language_modeling"
        )
        
        assert outputs['logits'].shape == (4, 16, 100)
    
    def test_multi_task_adaptation(self):
        """Test multi-task adaptation."""
        config = AdaptiveConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            use_task_adapters=True,
            use_domain_adapters=True,
        )
        
        model = SmartTransformer(config)
        
        input_ids = torch.randint(0, 100, (2, 8))
        attention_mask = torch.ones(2, 8)
        
        # Test different task types
        task_types = ["language_modeling", "classification", "generation"]
        domains = ["general", "medical", "technical"]
        
        for task_type in task_types:
            for domain in domains:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_type=task_type,
                    domain=domain
                )
                
                assert outputs['logits'].shape == (2, 8, 100)


if __name__ == "__main__":
    pytest.main([__file__]) 