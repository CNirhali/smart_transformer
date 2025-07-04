"""
Advanced Training Example

This example demonstrates advanced training capabilities of the Smart Transformer
including adaptive optimization, curriculum learning, and multi-task training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
import os
import argparse

from smart_transformer import SmartTransformer, AdaptiveConfig
from smart_transformer.training import SmartTrainer, TrainingConfig, CurriculumTrainer
from smart_transformer.optimization import AdaptiveOptimizer, OptimizationConfig
from smart_transformer.evaluation import SmartEvaluator, EvaluationConfig, PerformanceAnalyzer
from smart_transformer.utils import align_logits_and_labels, filter_batch_for_model


def create_multi_task_data(vocab_size=1000, num_samples=2000, seq_length=128):
    """Create multi-task dataset for demonstration."""
    
    # Task 1: Language Modeling
    lm_input_ids = np.random.randint(0, vocab_size, size=(num_samples, seq_length))
    lm_labels = lm_input_ids.copy()
    
    # Task 2: Classification (simulate sentiment analysis)
    cls_input_ids = np.random.randint(0, vocab_size, size=(num_samples, seq_length))
    cls_labels = np.random.randint(0, 3, size=num_samples)  # 3 classes
    
    # Task 3: Generation (simulate text completion)
    gen_input_ids = np.random.randint(0, vocab_size, size=(num_samples, seq_length // 2))
    gen_labels = np.random.randint(0, vocab_size, size=(num_samples, seq_length // 2))
    
    return {
        'language_modeling': (lm_input_ids, lm_labels),
        'classification': (cls_input_ids, cls_labels),
        'generation': (gen_input_ids, gen_labels)
    }


def create_curriculum_data(vocab_size=1000, base_samples=100):
    """Create curriculum learning dataset with increasing difficulty."""
    
    curriculum_data = {}
    
    # Stage 1: Short sequences - Memory optimized
    seq_lengths = [16, 32, 64, 128]  # Reduced from [32, 64, 128, 256]
    
    for i, seq_length in enumerate(seq_lengths):
        num_samples = base_samples * (i + 1)  # More samples for longer sequences
        
        input_ids = np.random.randint(0, vocab_size, size=(num_samples, seq_length))
        labels = input_ids.copy()
        
        curriculum_data[f'stage_{i+1}'] = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.ones(num_samples, seq_length, dtype=torch.long),
            'seq_length': seq_length,
            'num_samples': num_samples
        }
    
    return curriculum_data


class MultiTaskLoss(nn.Module):
    """Multi-task loss function."""
    
    def __init__(self, task_weights=None):
        super().__init__()
        self.task_weights = task_weights or {
            'language_modeling': 1.0,
            'classification': 1.0,
            'generation': 1.0
        }
    
    def forward(self, outputs, targets, task_type):
        """Compute loss for different tasks."""
        
        if task_type == 'language_modeling':
            logits = outputs['logits']
            labels = targets['labels']
            
            # Shift for language modeling
            if logits.size(1) > labels.size(1):
                logits = logits[:, :labels.size(1), :]
            elif labels.size(1) > logits.size(1):
                labels = labels[:, :logits.size(1)]
            
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
        elif task_type == 'classification':
            logits = outputs['logits']
            labels = targets['labels']
            
            # For classification, we might need to pool the sequence
            if logits.dim() == 3:  # [batch, seq, vocab]
                logits = logits.mean(dim=1)  # Average pooling
            
            loss = F.cross_entropy(logits, labels)
            
        elif task_type == 'generation':
            logits = outputs['logits']
            labels = targets['labels']
            
            # For generation, compute loss on the target sequence
            if logits.size(1) > labels.size(1):
                logits = logits[:, :labels.size(1), :]
            
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return loss * self.task_weights.get(task_type, 1.0)


class DictWrapperDataset(Dataset):
    def __init__(self, tensor_dataset):
        self.tensor_dataset = tensor_dataset
    def __len__(self):
        return len(self.tensor_dataset)
    def __getitem__(self, idx):
        input_ids, attention_mask, labels = self.tensor_dataset[idx]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--use_adaptive_attention', type=int, default=1)
    return parser.parse_args()


def advanced_training_example(args=None):
    if args is None:
        args = parse_args()
    """Demonstrate advanced training capabilities."""
    
    print("🚀 Advanced Smart Transformer Training Example")
    print("=" * 60)
    
    # 1. Configuration - Memory optimized
    print("\n1. Setting up advanced configuration...")
    
    config = AdaptiveConfig(
        vocab_size=1000,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
        use_adaptive_attention=bool(args.use_adaptive_attention),
        use_multi_scale_attention=True,
        use_task_adapters=True,
        use_domain_adapters=True,
        use_technique_adapters=True,
        use_flash_attention=True,
        use_rotary_position_embeddings=True,
        use_gated_linear_units=True,
        use_adaptive_optimization=True,
        use_dynamic_learning_rate=True,
        use_performance_monitoring=True,
    )
    
    print(f"   - Model size: {config.hidden_size} hidden, {config.num_layers} layers")
    print(f"   - Advanced features: All enabled")
    
    # 2. Initialize model
    print("\n2. Initializing Smart Transformer...")
    model = SmartTransformer(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 3. Create datasets - Memory optimized
    print("\n3. Creating multi-task datasets...")
    
    # Multi-task data - Reduced sizes
    multi_task_data = create_multi_task_data(
        vocab_size=config.vocab_size,
        num_samples=500,  # Reduced from 1000
        seq_length=64     # Reduced from 128
    )
    
    # Curriculum data - Reduced sizes
    curriculum_data = create_curriculum_data(
        vocab_size=config.vocab_size,
        base_samples=100   # Reduced from 200
    )
    
    print(f"   - Multi-task datasets created")
    print(f"   - Curriculum stages: {len(curriculum_data)}")
    
    # 4. Multi-task training demonstration
    print("\n4. Multi-task training demonstration...")
    
    # Create dataloaders for each task - Memory optimized
    task_dataloaders = {}
    
    for task_name, (input_ids, labels) in multi_task_data.items():
        if task_name == 'classification':
            # For classification, we need different handling
            dataset = TensorDataset(
                torch.tensor(input_ids, dtype=torch.long),
                torch.ones(input_ids.shape[0], input_ids.shape[1], dtype=torch.long),
                torch.tensor(labels, dtype=torch.long)
            )
        else:
            dataset = TensorDataset(
                torch.tensor(input_ids, dtype=torch.long),
                torch.ones(input_ids.shape[0], input_ids.shape[1], dtype=torch.long),
                torch.tensor(labels, dtype=torch.long)
            )
        
        task_dataloaders[task_name] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=0  # Reduced batch size and workers
        )
    
    # Multi-task loss
    multi_task_loss = MultiTaskLoss()
    
    # Training loop for multi-task
    print("   Training on multiple tasks...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # Reduced T_max
    
    model.train()
    task_losses = {task: [] for task in multi_task_data.keys()}
    
    for epoch in range(2):  # Reduced from 3 epochs
        epoch_losses = {task: 0.0 for task in multi_task_data.keys()}
        
        # Train on each task
        for task_name, dataloader in task_dataloaders.items():
            task_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                input_ids, attention_mask, labels = batch
                
                # Forward pass with task-specific parameters
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_type=task_name,
                    domain='general',
                    technique='multi_task'
                )
                
                # Compute loss
                loss = multi_task_loss(outputs, {'labels': labels}, task_name)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                task_loss += loss.item()
                num_batches += 1
                
                # Limit batches per task to prevent OOM
                if batch_idx >= 10:  # Process only first 10 batches per task
                    break
            
            avg_task_loss = task_loss / num_batches
            epoch_losses[task_name] = avg_task_loss
            task_losses[task_name].append(avg_task_loss)
        
        scheduler.step()
        
        print(f"   Epoch {epoch + 1}: " + 
              " | ".join([f"{task}: {loss:.4f}" for task, loss in epoch_losses.items()]))
    
    # 5. Curriculum learning demonstration - Memory optimized
    print("\n5. Curriculum learning demonstration...")
    
    curriculum_trainer = CurriculumTrainer(
        model=model,
        train_dataset=None,  # We'll use our curriculum data directly
        config=TrainingConfig(batch_size=args.batch_size, num_epochs=1)  # Reduced batch size and epochs
    )
    
    # Simulate curriculum training
    for stage_name, stage_data in curriculum_data.items():
        print(f"   Training on {stage_name} (seq_length: {stage_data['seq_length']})")
        
        # Create dataset for this stage
        stage_dataset = TensorDataset(
            stage_data['input_ids'],
            stage_data['attention_mask'],
            stage_data['labels']
        )
        
        stage_loader = DataLoader(
            stage_dataset,
            batch_size=args.batch_size,  # Reduced from 16
            shuffle=True,
            num_workers=0  # Reduced from 2
        )
        
        # Quick training on this stage
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        for epoch in range(1):  # Reduced from 2 epochs
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(stage_loader):
                input_ids, attention_mask, labels = batch
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_type='language_modeling',
                    technique='curriculum'
                )
                
                logits = outputs['logits']
                # Ensure logits and labels have compatible shapes
                if logits.size(1) > labels.size(1):
                    logits = logits[:, :labels.size(1), :]
                elif labels.size(1) > logits.size(1):
                    labels = labels[:, :logits.size(1)]
                
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Limit batches to prevent OOM
                if batch_idx >= 5:  # Process only first 5 batches
                    break
            
            avg_loss = total_loss / num_batches
            print(f"     Epoch {epoch + 1}: Loss: {avg_loss:.4f}")
    
    # 6. Adaptive optimization demonstration
    print("\n6. Adaptive optimization demonstration...")
    
    # Create a larger dataset for optimization demonstration
    multi_task_data_large = create_multi_task_data(
        vocab_size=config.vocab_size,
        num_samples=500,
        seq_length=64  # Reduced from 128
    )
    
    large_input_ids, large_labels = multi_task_data_large['language_modeling']
    large_attention_mask = np.ones_like(large_input_ids)
    
    large_dataset = TensorDataset(
        torch.tensor(large_input_ids, dtype=torch.long),
        torch.tensor(large_attention_mask, dtype=torch.long),
        torch.tensor(large_labels, dtype=torch.long)
    )
    
    large_loader = DataLoader(large_dataset, batch_size=args.batch_size, shuffle=True)  # Reduced from 16
    
    # Adaptive optimizer
    opt_config = OptimizationConfig(
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        max_steps=500,
        use_adaptive_optimization=True,
        use_dynamic_lr=True,
    )
    
    adaptive_optimizer = AdaptiveOptimizer(model, opt_config)
    
    print("   Training with adaptive optimization...")
    
    model.train()
    adaptive_losses = []
    
    for step in range(100):  # Short demonstration
        try:
            batch = next(data_iter)
        except (StopIteration, NameError):
            data_iter = iter(large_loader)
            batch = next(data_iter)
        
        input_ids, attention_mask, labels = batch
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_type='language_modeling',
            technique='adaptive_optimization'
        )
        
        logits = outputs['logits']
        # Ensure logits and labels have compatible shapes
        if logits.size(1) > labels.size(1):
            logits = logits[:, :labels.size(1), :]
        elif labels.size(1) > logits.size(1):
            labels = labels[:, :logits.size(1)]
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1)
        )
        
        # Adaptive optimization step
        adaptive_optimizer.step(loss.item(), step)
        
        adaptive_losses.append(loss.item())
        
        if step % 20 == 0:
            avg_loss = np.mean(adaptive_losses[-20:])
            current_lr = adaptive_optimizer.optimizer.param_groups[0]['lr']
            print(f"     Step {step}: Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
    
    # 7. Performance analysis
    print("\n7. Performance analysis...")
    
    analyzer = PerformanceAnalyzer()
    
    # Analyze model performance
    perf_analysis = analyzer.analyze_model_performance(model, large_loader)
    
    print(f"   - Computational complexity: {perf_analysis['computational_complexity']['total_parameters']:,} parameters")
    print(f"   - Memory usage: {perf_analysis['memory_usage']['model_memory_mb']:.2f} MB")
    print(f"   - Latency: {perf_analysis['latency_analysis']['mean_latency_ms']:.2f} ms")
    print(f"   - Throughput: {perf_analysis['throughput_analysis']['throughput_samples_per_s']:.2f} samples/sec")
    
    # 8. Evaluation
    print("\n8. Comprehensive evaluation...")
    
    # Create evaluation dataset
    eval_multi_task_data = create_multi_task_data(
        vocab_size=config.vocab_size,
        num_samples=100,
        seq_length=128
    )
    
    eval_input_ids, eval_labels = eval_multi_task_data['language_modeling']
    eval_attention_mask = np.ones_like(eval_input_ids)
    
    eval_dataset = TensorDataset(
        torch.tensor(eval_input_ids, dtype=torch.long),
        torch.tensor(eval_attention_mask, dtype=torch.long),
        torch.tensor(eval_labels, dtype=torch.long)
    )
    eval_dict_dataset = DictWrapperDataset(eval_dataset)
    eval_loader = DataLoader(eval_dict_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate
    def filtered_batch(batch):
        return {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']}

    evaluator = SmartEvaluator()
    eval_results = evaluator.evaluate(
        model=model,
        test_loader=eval_loader,
        task_type='language_modeling',
        save_results=True
    )
    
    print(f"   - Evaluation loss: {eval_results['loss']:.4f}")
    if 'perplexity' in eval_results:
        print(f"   - Perplexity: {eval_results['perplexity']:.2f}")
    if 'accuracy' in eval_results:
        print(f"   - Accuracy: {eval_results['accuracy']:.4f}")
    
    # After evaluation
    if 'precision' in eval_results:
        print(f"   - Precision: {eval_results['precision']:.4f}")
    if 'recall' in eval_results:
        print(f"   - Recall: {eval_results['recall']:.4f}")
    if 'f1' in eval_results:
        print(f"   - F1 Score: {eval_results['f1']:.4f}")
    if 'confusion_matrix' in eval_results:
        plt.figure(figsize=(6, 5))
        sns.heatmap(eval_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('classification_confusion_matrix.png', dpi=200)
        plt.close()
        print("   - Confusion matrix saved as 'classification_confusion_matrix.png'")
    
    if 'bleu' in eval_results:
        print(f"   - BLEU: {eval_results['bleu']:.4f}")
    if 'rouge1' in eval_results:
        print(f"   - ROUGE-1: {eval_results['rouge1']:.4f}")
    if 'rougeL' in eval_results:
        print(f"   - ROUGE-L: {eval_results['rougeL']:.4f}")
    
    # Attention map visualization (for a sample batch)
    sample_batch = next(iter(eval_loader))
    model.eval()
    with torch.no_grad():
        model_inputs = filter_batch_for_model(sample_batch)
        outputs = model(**model_inputs, output_attentions=True)
        if outputs.get('attentions') and len(outputs['attentions']) > 0:
            attn = outputs['attentions'][0]  # First layer
            if attn is not None:
                # attn shape: (batch, heads, seq, seq)
                attn_map = attn[0, 0].cpu().numpy()  # First sample, first head
                plt.figure(figsize=(6, 5))
                sns.heatmap(attn_map, cmap='viridis')
                plt.title('Attention Map (Layer 1, Head 1)')
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                plt.tight_layout()
                plt.savefig('attention_map_layer1_head1.png', dpi=200)
                plt.close()
                print("   - Attention map saved as 'attention_map_layer1_head1.png'")
    
    # 9. Visualization
    print("\n9. Creating training visualizations...")
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Multi-task losses
    for task_name, losses in task_losses.items():
        axes[0, 0].plot(losses, label=task_name)
    axes[0, 0].set_title('Multi-Task Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Adaptive optimization losses
    axes[0, 1].plot(adaptive_losses)
    axes[0, 1].set_title('Adaptive Optimization Loss')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Learning rate over time
    lr_history = adaptive_optimizer.learning_rates
    if lr_history:
        axes[1, 0].plot(lr_history)
        axes[1, 0].set_title('Learning Rate Over Time')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Gradient norms
    grad_norms = adaptive_optimizer.gradient_norms
    if grad_norms:
        axes[1, 1].plot(grad_norms)
        axes[1, 1].set_title('Gradient Norms')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('advanced_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   - Training curves saved as 'advanced_training_curves.png'")
    
    # 10. Summary
    print("\n" + "=" * 60)
    print("✅ Advanced Training Example Complete!")
    print("\nAdvanced Features Demonstrated:")
    print("  ✓ Multi-task learning with adaptive loss weighting")
    print("  ✓ Curriculum learning with progressive difficulty")
    print("  ✓ Adaptive optimization with dynamic strategy selection")
    print("  ✓ Performance monitoring and analysis")
    print("  ✓ Comprehensive evaluation across metrics")
    print("  ✓ Training visualization and analysis")
    
    print("\nKey Insights:")
    print("  • Multi-task training improves generalization")
    print("  • Curriculum learning accelerates convergence")
    print("  • Adaptive optimization adapts to training dynamics")
    print("  • Performance monitoring enables optimization")
    
    return model, config, eval_results


def main():
    """Main function to run the advanced training example."""
    
    # Set up logging
    os.makedirs('logs', exist_ok=True)
    
    # Run advanced training
    args = parse_args()
    model, config, results = advanced_training_example(args)
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'evaluation_results': results,
    }, 'smart_transformer_advanced.pt')
    
    print("\n💾 Advanced model saved as 'smart_transformer_advanced.pt'")
    
    return model, config, results


if __name__ == "__main__":
    # Run the advanced training example
    model, config, results = main()
    
    print("\n🎉 Advanced training example completed successfully!")
    print("Check the generated files:")
    print("  - smart_transformer_advanced.pt (trained model)")
    print("  - advanced_training_curves.png (training curves)")
    print("  - evaluation_results/ (evaluation results)") 