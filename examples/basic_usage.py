"""
Basic Usage Example

This example demonstrates how to use the Smart Transformer for
various NLP tasks with adaptive capabilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transformers import AutoTokenizer

from smart_transformer import SmartTransformer, AdaptiveConfig
from smart_transformer.training import SmartTrainer, TrainingConfig
from smart_transformer.evaluation import SmartEvaluator, EvaluationConfig


def create_sample_data(vocab_size=1000, num_samples=1000, seq_length=128):
    """Create sample data for demonstration."""
    
    # Generate random token sequences
    input_ids = np.random.randint(0, vocab_size, size=(num_samples, seq_length))
    labels = input_ids.copy()
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Create attention masks
    attention_mask = torch.ones_like(input_ids)
    
    return input_ids, attention_mask, labels


def main():
    """Main function demonstrating Smart Transformer usage."""
    
    print("🚀 Smart Transformer Basic Usage Example")
    print("=" * 50)
    
    # 1. Configuration
    print("\n1. Setting up configuration...")
    config = AdaptiveConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
        use_adaptive_attention=True,
        use_multi_scale_attention=True,
        use_task_adapters=True,
        use_domain_adapters=True,
        use_technique_adapters=True,
        use_flash_attention=True,
        use_rotary_position_embeddings=True,
        use_gated_linear_units=True,
        use_adaptive_optimization=True,
        use_dynamic_learning_rate=True,
    )
    
    print(f"   - Hidden size: {config.hidden_size}")
    print(f"   - Number of layers: {config.num_layers}")
    print(f"   - Attention heads: {config.num_attention_heads}")
    print(f"   - Adaptive features: {config.use_adaptive_attention}")
    
    # 2. Initialize Smart Transformer
    print("\n2. Initializing Smart Transformer...")
    model = SmartTransformer(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 3. Create sample data
    print("\n3. Creating sample data...")
    input_ids, attention_mask, labels = create_sample_data(
        vocab_size=config.vocab_size,
        num_samples=100,
        seq_length=64
    )
    
    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"   - Dataset size: {len(dataset)}")
    print(f"   - Batch size: 8")
    print(f"   - Sequence length: 64")
    
    # 4. Test forward pass
    print("\n4. Testing forward pass...")
    model.eval()
    
    with torch.no_grad():
        # Test with different task types
        task_types = ["language_modeling", "classification", "generation"]
        
        for task_type in task_types:
            print(f"   Testing {task_type}...")
            
            batch = {
                'input_ids': input_ids[:2],
                'attention_mask': attention_mask[:2],
                'task_type': task_type,
                'domain': 'general',
                'technique': 'standard'
            }
            
            outputs = model(**batch)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
                print(f"     - Output shape: {logits.shape}")
                print(f"     - Output dtype: {logits.dtype}")
            else:
                print(f"     - Output shape: {outputs.shape}")
                print(f"     - Output dtype: {outputs.dtype}")
    
    # 5. Test text generation
    print("\n5. Testing text generation...")
    
    # Create a simple prompt
    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # Simple sequence
    
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt,
            max_length=20,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=0,
            eos_token_id=2
        )
    
    print(f"   - Input prompt: {prompt[0].tolist()}")
    print(f"   - Generated sequence: {generated[0].tolist()}")
    
    # 6. Demonstrate adaptive features
    print("\n6. Demonstrating adaptive features...")
    
    # Test with different domains
    domains = ["general", "medical", "technical", "academic"]
    
    for domain in domains:
        print(f"   Testing domain: {domain}")
        
        batch = {
            'input_ids': input_ids[:1],
            'attention_mask': attention_mask[:1],
            'domain': domain
        }
        
        with torch.no_grad():
            outputs = model(**batch)
            if isinstance(outputs, dict):
                print(f"     - Output shape: {outputs['logits'].shape}")
    
    # 7. Performance analysis
    print("\n7. Performance analysis...")
    
    from smart_transformer.evaluation import PerformanceAnalyzer
    
    analyzer = PerformanceAnalyzer()
    
    # Analyze model performance
    perf_analysis = analyzer.analyze_model_performance(model, dataloader)
    
    print(f"   - Total parameters: {perf_analysis['computational_complexity']['total_parameters']:,}")
    print(f"   - Model memory: {perf_analysis['memory_usage']['model_memory_mb']:.2f} MB")
    print(f"   - Mean latency: {perf_analysis['latency_analysis']['mean_latency_ms']:.2f} ms")
    print(f"   - Throughput: {perf_analysis['throughput_analysis']['throughput_samples_per_s']:.2f} samples/sec")
    
    # 8. Training setup (demonstration)
    print("\n8. Training setup demonstration...")
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=8,
        num_epochs=2,
        learning_rate=1e-4,
        use_adaptive_training=True,
        use_early_stopping=True,
        use_mixed_precision=True,
    )
    
    print(f"   - Training epochs: {training_config.num_epochs}")
    print(f"   - Learning rate: {training_config.learning_rate}")
    print(f"   - Adaptive training: {training_config.use_adaptive_training}")
    
    # Note: Actual training would require more data and time
    print("   - Note: Skipping actual training for demonstration")
    
    # 9. Evaluation setup
    print("\n9. Evaluation setup...")
    
    # Create evaluation dataset
    eval_input_ids, eval_attention_mask, eval_labels = create_sample_data(
        vocab_size=config.vocab_size,
        num_samples=50,
        seq_length=64
    )
    
    eval_dataset = TensorDataset(eval_input_ids, eval_attention_mask, eval_labels)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False)
    
    # Evaluation configuration
    eval_config = EvaluationConfig(
        batch_size=8,
        compute_perplexity=True,
        compute_accuracy=True,
        save_predictions=True,
    )
    
    evaluator = SmartEvaluator(eval_config)
    
    print(f"   - Evaluation samples: {len(eval_dataset)}")
    print(f"   - Computing perplexity: {eval_config.compute_perplexity}")
    print(f"   - Computing accuracy: {eval_config.compute_accuracy}")
    
    # 10. Summary
    print("\n" + "=" * 50)
    print("✅ Smart Transformer Basic Usage Complete!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Adaptive attention mechanisms")
    print("  ✓ Multi-scale processing")
    print("  ✓ Task-specific adapters")
    print("  ✓ Domain-specific adaptations")
    print("  ✓ Dynamic optimization")
    print("  ✓ Performance monitoring")
    print("  ✓ Text generation capabilities")
    print("  ✓ Comprehensive evaluation")
    
    print("\nNext Steps:")
    print("  1. Train on your specific dataset")
    print("  2. Fine-tune for your task")
    print("  3. Deploy in production")
    print("  4. Monitor performance")
    
    return model, config


if __name__ == "__main__":
    # Run the example
    model, config = main()
    
    # Save the model for later use
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, 'smart_transformer_example.pt')
    
    print("\n💾 Model saved as 'smart_transformer_example.pt'") 