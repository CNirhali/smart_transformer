"""
Smart Evaluation

This module implements comprehensive evaluation strategies for assessing
model performance across different metrics and tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import time
import json
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from smart_transformer.utils import align_logits_and_labels, filter_batch_for_model
import nltk
from nltk.translate.bleu_score import corpus_bleu
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    batch_size: int = 16
    max_length: int = 512
    use_gpu: bool = True
    compute_perplexity: bool = True
    compute_accuracy: bool = True
    compute_bleu: bool = True
    compute_rouge: bool = True
    save_predictions: bool = True
    output_dir: str = "evaluation_results"


class SmartEvaluator:
    """
    Smart evaluator that provides comprehensive model assessment
    across multiple metrics and tasks.
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.device = torch.device('cuda' if self.config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Evaluation metrics
        self.metrics = {
            'perplexity': self._compute_perplexity,
            'accuracy': self._compute_accuracy,
            'precision': self._compute_precision,
            'recall': self._compute_recall,
            'f1': self._compute_f1,
            'bleu': self._compute_bleu,
            'rouge': self._compute_rouge,
            'latency': self._compute_latency,
            'throughput': self._compute_throughput,
        }
        
        # Results storage
        self.results = {}
        
    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        task_type: str = "language_modeling",
        save_results: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: The model to evaluate
            test_loader: Test data loader
            task_type: Type of task (language_modeling, classification, etc.)
            save_results: Whether to save results to file
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        print(f"Starting evaluation for {task_type}...")
        
        model.eval()
        model.to(self.device)
        
        # Initialize results
        self.results = {
            'task_type': task_type,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {},
            'predictions': [],
            'ground_truth': [],
        }
        
        # Evaluation loop
        total_loss = 0
        num_batches = 0
        start_time = time.time()
        
        all_outputs = []
        all_labels = []
        all_logits = []
        all_labels_for_logits = []
        
        with torch.no_grad():
            for batch in test_loader:
                model_inputs = filter_batch_for_model(batch)
                outputs = model(**model_inputs) if isinstance(model_inputs, dict) else model(model_inputs)
                total_loss += self._compute_loss(outputs, batch, task_type).item()
                
                # Store predictions and ground truth
                self._store_predictions(outputs, batch, task_type)
                
                all_outputs.append(outputs)
                if isinstance(batch, dict) and 'labels' in batch:
                    all_labels.append(batch['labels'])
                    if 'logits' in outputs:
                        all_logits.append(outputs['logits'])
                        all_labels_for_logits.append(batch['labels'])
                
                num_batches += 1
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        self.results['metrics']['loss'] = avg_loss
        
        # Task-specific metrics
        if task_type == "language_modeling":
            self._evaluate_language_modeling()
        elif task_type == "classification":
            self._evaluate_classification()
        elif task_type == "generation":
            self._evaluate_generation()
        
        # Performance metrics
        self._evaluate_performance(start_time, num_batches)
        
        # Save results
        if save_results:
            self._save_results()
        
        print("Evaluation completed!")
        
        # Always set these attributes before metrics
        self.all_logits = all_logits
        self.all_labels = all_labels_for_logits
        # Fallback: ensure attributes exist
        if not hasattr(self, 'all_logits'):
            self.all_logits = []
        if not hasattr(self, 'all_labels'):
            self.all_labels = []
        
        return self.results['metrics']
    
    def _compute_loss(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        batch: Dict[str, torch.Tensor],
        task_type: str
    ) -> torch.Tensor:
        """Compute loss based on task type."""
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        if task_type == "language_modeling":
            # Language modeling loss
            if 'labels' in batch:
                labels = batch['labels']
            else:
                labels = batch['input_ids'][:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()
            
            # Align sequence length
            if logits.size(1) > labels.size(1):
                logits = logits[:, :labels.size(1), :]
            elif labels.size(1) > logits.size(1):
                labels = labels[:, :logits.size(1)]
            
            logits, labels = align_logits_and_labels(logits, labels)
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        
        elif task_type == "classification":
            # Classification loss
            labels = batch['labels']
            return F.cross_entropy(logits, labels)
        
        else:
            # Default to cross-entropy
            labels = batch.get('labels', batch.get('input_ids', None))
            if labels is not None:
                logits, labels = align_logits_and_labels(logits, labels)
                return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                return torch.tensor(0.0, device=self.device)
    
    def _store_predictions(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        batch: Dict[str, torch.Tensor],
        task_type: str
    ):
        """Store model predictions and ground truth."""
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Store predictions
        if task_type == "classification":
            predictions = torch.argmax(logits, dim=-1)
            self.results['predictions'].extend(predictions.cpu().numpy().tolist())
            
            if 'labels' in batch:
                self.results['ground_truth'].extend(batch['labels'].cpu().numpy().tolist())
        
        elif task_type == "language_modeling":
            # For language modeling, store perplexity-related data
            if 'labels' in batch:
                labels = batch['labels']
            else:
                labels = batch['input_ids'][:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()
            
            # Store token-level predictions for analysis
            token_predictions = torch.argmax(logits, dim=-1)
            self.results['predictions'].extend(token_predictions.cpu().numpy().tolist())
            self.results['ground_truth'].extend(labels.cpu().numpy().tolist())
    
    def _evaluate_language_modeling(self):
        """Evaluate language modeling performance."""
        
        if self.config.compute_perplexity:
            perplexity = self._compute_perplexity()
            self.results['metrics']['perplexity'] = perplexity
        
        if self.config.compute_accuracy:
            accuracy = self._compute_accuracy()
            self.results['metrics']['accuracy'] = accuracy
    
    def _evaluate_classification(self):
        """Evaluate classification performance."""
        
        if len(self.results['predictions']) > 0 and len(self.results['ground_truth']) > 0:
            predictions = np.array(self.results['predictions'])
            ground_truth = np.array(self.results['ground_truth'])
            
            # Compute classification metrics
            accuracy = accuracy_score(ground_truth, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truth, predictions, average='weighted'
            )
            
            self.results['metrics']['accuracy'] = accuracy
            self.results['metrics']['precision'] = precision
            self.results['metrics']['recall'] = recall
            self.results['metrics']['f1'] = f1
            
            # Confusion matrix
            cm = confusion_matrix(ground_truth, predictions)
            self.results['confusion_matrix'] = cm.tolist()
    
    def _evaluate_generation(self):
        """Evaluate text generation performance."""
        
        if self.config.compute_bleu:
            bleu_score = self._compute_bleu()
            self.results['metrics']['bleu'] = bleu_score
        
        if self.config.compute_rouge:
            rouge_score = self._compute_rouge()
            self.results['metrics']['rouge'] = rouge_score
    
    def _evaluate_performance(self, start_time: float, num_batches: int):
        """Evaluate computational performance."""
        
        total_time = time.time() - start_time
        avg_time_per_batch = total_time / num_batches
        
        self.results['metrics']['total_time'] = total_time
        self.results['metrics']['avg_time_per_batch'] = avg_time_per_batch
        self.results['metrics']['throughput'] = num_batches / total_time
    
    def _compute_perplexity(self):
        ce_loss = 0.0
        total_tokens = 0
        for logits, labels in zip(self.all_logits, self.all_labels):
            logits, labels = align_logits_and_labels(logits, labels)
            ce_loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction='sum').item()
            total_tokens += labels.numel()
        return np.exp(ce_loss / total_tokens) if total_tokens > 0 else float('inf')
    
    def _compute_accuracy(self) -> float:
        """Compute accuracy."""
        
        if len(self.results['predictions']) == 0 or len(self.results['ground_truth']) == 0:
            return 0.0
        
        predictions = np.array(self.results['predictions'])
        ground_truth = np.array(self.results['ground_truth'])
        
        # Flatten if needed
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if ground_truth.ndim > 1:
            ground_truth = ground_truth.flatten()
        
        return accuracy_score(ground_truth, predictions)
    
    def _compute_precision(self) -> float:
        """Compute precision."""
        # Implementation depends on task type
        return 0.0
    
    def _compute_recall(self) -> float:
        """Compute recall."""
        # Implementation depends on task type
        return 0.0
    
    def _compute_f1(self) -> float:
        """Compute F1 score."""
        # Implementation depends on task type
        return 0.0
    
    def _compute_bleu(self) -> float:
        """Compute BLEU score."""
        # Collect references and hypotheses
        references = []
        hypotheses = []
        for outputs, batch in zip(self.all_outputs, self.all_labels):
            # Assume outputs['sequences'] or outputs['logits']
            if 'sequences' in outputs:
                pred_seq = outputs['sequences']
            else:
                pred_seq = outputs['logits'].argmax(dim=-1)
            if hasattr(pred_seq, 'cpu'):
                pred_seq = pred_seq.cpu().numpy()
            if hasattr(batch, 'cpu'):
                ref_seq = batch.cpu().numpy()
            else:
                ref_seq = batch
            hypotheses.extend([list(map(str, h)) for h in pred_seq])
            references.extend([[list(map(str, r))] for r in ref_seq])
        # BLEU
        bleu = corpus_bleu(references, hypotheses)
        return bleu
    
    def _compute_rouge(self) -> float:
        """Compute ROUGE score."""
        # ROUGE (if available)
        if ROUGE_AVAILABLE:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge1s, rougels = [], []
            for ref, hyp in zip(references, hypotheses):
                ref_str = ' '.join(ref[0])
                hyp_str = ' '.join(hyp)
                scores = scorer.score(ref_str, hyp_str)
                rouge1s.append(scores['rouge1'].fmeasure)
                rougels.append(scores['rougeL'].fmeasure)
            rouge1 = sum(rouge1s) / len(rouge1s) if rouge1s else 0.0
            rougeL = sum(rougels) / len(rougels) if rougels else 0.0
            return rouge1
        else:
            return 0.0
    
    def _compute_latency(self) -> float:
        """Compute average inference latency."""
        return self.results['metrics'].get('avg_time_per_batch', 0.0)
    
    def _compute_throughput(self) -> float:
        """Compute throughput (samples per second)."""
        return self.results['metrics'].get('throughput', 0.0)
    
    def _save_results(self):
        """Save evaluation results to file."""
        
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save metrics
        results_file = os.path.join(self.config.output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save confusion matrix plot if available
        if 'confusion_matrix' in self.results:
            self._plot_confusion_matrix()
        
        print(f"Results saved to {self.config.output_dir}")
    
    def _plot_confusion_matrix(self):
        """Plot and save confusion matrix."""
        
        cm = np.array(self.results['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_file = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _compute_classification_metrics(self):
        # Collect predictions and ground truth
        y_true = []
        y_pred = []
        for outputs, batch in zip(self.all_outputs, self.all_labels):
            # Assume outputs['logits'] shape: [batch, num_classes] or [batch, seq, num_classes]
            logits = outputs['logits']
            if logits.dim() == 3:
                logits = logits.mean(dim=1)  # Pool over sequence
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch.cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        self.results['metrics']['precision'] = precision
        self.results['metrics']['recall'] = recall
        self.results['metrics']['f1'] = f1
        self.results['metrics']['confusion_matrix'] = cm


class PerformanceAnalyzer:
    """
    Performance analyzer that provides detailed analysis of model performance
    across different dimensions.
    """
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_model_performance(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model performance analysis.
        
        Args:
            model: The model to analyze
            test_loader: Test data loader
            device: Device to run analysis on
            
        Returns:
            Dictionary containing performance analysis
        """
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.eval()
        model.to(device)
        
        # Initialize analysis
        self.analysis_results = {
            'computational_complexity': {},
            'memory_usage': {},
            'latency_analysis': {},
            'throughput_analysis': {},
            'scalability_analysis': {},
        }
        
        # Analyze computational complexity
        self._analyze_computational_complexity(model)
        
        # Analyze memory usage
        self._analyze_memory_usage(model, device)
        
        # Analyze latency
        self._analyze_latency(model, test_loader, device)
        
        # Analyze throughput
        self._analyze_throughput(model, test_loader, device)
        
        # Analyze scalability
        self._analyze_scalability(model, test_loader, device)
        
        return self.analysis_results
    
    def _analyze_computational_complexity(self, model: nn.Module):
        """Analyze computational complexity of the model."""
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count operations (simplified)
        total_ops = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_ops += module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                total_ops += module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]
        
        self.analysis_results['computational_complexity'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_operations': total_ops,
            'parameter_efficiency': trainable_params / total_params if total_params > 0 else 0,
        }
    
    def _analyze_memory_usage(self, model: nn.Module, device: torch.device):
        """Analyze memory usage of the model."""
        
        # Model memory
        model_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # GPU memory if available
        gpu_memory = 0
        if device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated(device)
        
        self.analysis_results['memory_usage'] = {
            'model_memory_mb': model_memory / (1024 * 1024),
            'gpu_memory_mb': gpu_memory / (1024 * 1024) if gpu_memory > 0 else 0,
            'memory_efficiency': model_memory / (1024 * 1024 * 1024),  # GB
        }
    
    def _analyze_latency(self, model: nn.Module, test_loader: DataLoader, device: torch.device):
        """Analyze inference latency."""
        model.eval()
        latencies = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    # Convert list/tuple to expected format
                    if len(batch) >= 1:
                        input_ids = batch[0].to(device) if torch.is_tensor(batch[0]) else torch.tensor(batch[0]).to(device)
                        batch = {'input_ids': input_ids}
                    else:
                        continue
                else:
                    batch = batch.to(device)
                
                # Measure forward pass time
                start_time = time.time()
                _ = model(**batch) if isinstance(batch, dict) else model(batch)
                end_time = time.time()
                
                latencies.append(end_time - start_time)
                
                if len(latencies) >= 10:  # Limit to 10 samples for speed
                    break
        
        latencies = np.array(latencies)
        
        self.analysis_results['latency_analysis'] = {
            'mean_latency_ms': np.mean(latencies) * 1000,
            'median_latency_ms': np.median(latencies) * 1000,
            'std_latency_ms': np.std(latencies) * 1000,
            'min_latency_ms': np.min(latencies) * 1000,
            'max_latency_ms': np.max(latencies) * 1000,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000,
        }
    
    def _analyze_throughput(self, model: nn.Module, test_loader: DataLoader, device: torch.device):
        """Analyze model throughput."""
        model.eval()
        num_samples = 0
        start_time = time.time()
        with torch.no_grad():
            for batch in test_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    # Convert list/tuple to expected format
                    if len(batch) >= 1:
                        input_ids = batch[0].to(device) if torch.is_tensor(batch[0]) else torch.tensor(batch[0]).to(device)
                        batch = {'input_ids': input_ids}
                    else:
                        continue
                else:
                    batch = batch.to(device)
                
                # Forward pass
                _ = model(**batch) if isinstance(batch, dict) else model(batch)
                num_samples += batch['input_ids'].size(0) if isinstance(batch, dict) and 'input_ids' in batch else 1
                if num_samples >= 100:
                    break
        end_time = time.time()
        elapsed = end_time - start_time
        throughput = num_samples / elapsed if elapsed > 0 else 0
        self.analysis_results['throughput_analysis'] = {
            'num_samples': num_samples,
            'elapsed_time_s': elapsed,
            'throughput_samples_per_s': throughput,
        }
    
    def _analyze_scalability(self, model, test_loader, device):
        """Analyze model scalability with increasing batch sizes."""
        model.eval()
        batch_sizes = [1, 2, 4, 8, 16]
        scalability_results = {}
        for batch_size in batch_sizes:
            latencies = []
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    # Handle different batch formats
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    elif isinstance(batch, (list, tuple)):
                        # Convert list/tuple to expected format
                        if len(batch) >= 1:
                            input_ids = batch[0].to(device) if torch.is_tensor(batch[0]) else torch.tensor(batch[0]).to(device)
                            batch = {'input_ids': input_ids}
                        else:
                            continue
                    else:
                        batch = batch.to(device)
                    # Adjust batch size if needed
                    if isinstance(batch, dict) and 'input_ids' in batch:
                        if batch['input_ids'].size(0) != batch_size:
                            continue
                    else:
                        continue
                    start_time = time.time()
                    _ = model(**batch) if isinstance(batch, dict) else model(batch)
                    end_time = time.time()
                    latencies.append(end_time - start_time)
                    if len(latencies) >= 5:
                        break
            if latencies:
                scalability_results[batch_size] = {
                    'mean_latency_ms': np.mean(latencies) * 1000,
                    'min_latency_ms': np.min(latencies) * 1000,
                    'max_latency_ms': np.max(latencies) * 1000,
                }
        self.analysis_results['scalability_analysis'] = scalability_results
    
    def generate_performance_report(self, output_file: str = "performance_report.json"):
        """Generate a comprehensive performance report."""
        
        report = {
            'summary': {
                'total_parameters': self.analysis_results['computational_complexity']['total_parameters'],
                'model_memory_mb': self.analysis_results['memory_usage']['model_memory_mb'],
                'mean_latency_ms': self.analysis_results['latency_analysis']['mean_latency_ms'],
                'throughput_sps': self.analysis_results['throughput_analysis']['throughput_samples_per_s'],
            },
            'detailed_analysis': self.analysis_results,
            'recommendations': self._generate_recommendations(),
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Performance report saved to {output_file}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        
        recommendations = []
        
        # Analyze results and generate recommendations
        if self.analysis_results['latency_analysis']['mean_latency_ms'] > 100:
            recommendations.append("Consider model optimization techniques like quantization or pruning")
        
        if self.analysis_results['memory_usage']['model_memory_mb'] > 1000:
            recommendations.append("Model is large, consider using gradient checkpointing or model parallelism")
        
        if self.analysis_results['computational_complexity']['parameter_efficiency'] < 0.8:
            recommendations.append("Low parameter efficiency, consider removing unused parameters")
        
        # Scalability recommendations
        scalability = self.analysis_results['scalability_analysis']
        if len(scalability) > 1:
            batch_sizes = list(scalability.keys())
            efficiencies = [scalability[bs]['efficiency'] for bs in batch_sizes]
            
            if max(efficiencies) / min(efficiencies) > 2:
                recommendations.append("High efficiency variance across batch sizes, optimize batch processing")
        
        return recommendations


class ModelComparison:
    """
    Model comparison utility for comparing multiple models across
    different metrics and tasks.
    """
    
    def __init__(self):
        self.comparison_results = {}
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        test_loader: DataLoader,
        evaluator: SmartEvaluator = None,
        analyzer: PerformanceAnalyzer = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models across different metrics.
        
        Args:
            models: Dictionary of model names to model instances
            test_loader: Test data loader
            evaluator: SmartEvaluator instance
            analyzer: PerformanceAnalyzer instance
            
        Returns:
            Dictionary containing comparison results
        """
        
        if evaluator is None:
            evaluator = SmartEvaluator()
        
        if analyzer is None:
            analyzer = PerformanceAnalyzer()
        
        comparison_results = {
            'models': {},
            'summary': {},
            'rankings': {},
        }
        
        # Evaluate each model
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            
            # Evaluation metrics
            eval_metrics = evaluator.evaluate(model, test_loader, save_results=False)
            
            # Performance analysis
            perf_analysis = analyzer.analyze_model_performance(model, test_loader)
            
            comparison_results['models'][model_name] = {
                'evaluation': eval_metrics,
                'performance': perf_analysis,
            }
        
        # Generate rankings
        comparison_results['rankings'] = self._generate_rankings(comparison_results['models'])
        
        # Generate summary
        comparison_results['summary'] = self._generate_summary(comparison_results['models'])
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def _generate_rankings(self, models_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate rankings for different metrics."""
        
        rankings = {}
        
        # Ranking metrics
        ranking_metrics = [
            'accuracy', 'perplexity', 'f1', 'throughput', 'mean_latency_ms'
        ]
        
        for metric in ranking_metrics:
            model_scores = {}
            
            for model_name, results in models_results.items():
                # Try to get metric from evaluation results
                score = results['evaluation'].get(metric, None)
                
                # If not in evaluation, try performance analysis
                if score is None:
                    if metric == 'throughput':
                        score = results['performance']['throughput_analysis']['throughput_samples_per_s']
                    elif metric == 'mean_latency_ms':
                        score = results['performance']['latency_analysis']['mean_latency_ms']
                
                if score is not None:
                    model_scores[model_name] = score
            
            # Sort models by score (higher is better for most metrics)
            if metric == 'perplexity' or metric == 'mean_latency_ms':
                # Lower is better for these metrics
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
            else:
                # Higher is better for other metrics
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            rankings[metric] = [model_name for model_name, _ in sorted_models]
        
        return rankings
    
    def _generate_summary(self, models_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        
        summary = {
            'best_models': {},
            'average_metrics': {},
            'metric_ranges': {},
        }
        
        # Find best models for each metric
        for metric in ['accuracy', 'perplexity', 'f1', 'throughput']:
            best_model = None
            best_score = None
            
            for model_name, results in models_results.items():
                score = results['evaluation'].get(metric, None)
                if score is not None:
                    if best_score is None:
                        best_score = score
                        best_model = model_name
                    elif metric == 'perplexity':
                        if score < best_score:
                            best_score = score
                            best_model = model_name
                    else:
                        if score > best_score:
                            best_score = score
                            best_model = model_name
            
            if best_model is not None:
                summary['best_models'][metric] = {
                    'model': best_model,
                    'score': best_score
                }
        
        return summary
    
    def save_comparison_report(self, output_file: str = "model_comparison_report.json"):
        """Save comparison report to file."""
        
        with open(output_file, 'w') as f:
            json.dump(self.comparison_results, f, indent=2)
        
        print(f"Comparison report saved to {output_file}")
    
    def plot_comparison(self, output_file: str = "model_comparison.png"):
        """Generate comparison plots."""
        
        if not self.comparison_results:
            print("No comparison results available. Run compare_models first.")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Accuracy comparison
        self._plot_metric_comparison('accuracy', axes[0, 0])
        
        # Plot 2: Perplexity comparison
        self._plot_metric_comparison('perplexity', axes[0, 1])
        
        # Plot 3: Throughput comparison
        self._plot_metric_comparison('throughput', axes[1, 0])
        
        # Plot 4: Latency comparison
        self._plot_metric_comparison('mean_latency_ms', axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to {output_file}")
    
    def _plot_metric_comparison(self, metric: str, ax):
        """Plot comparison for a specific metric."""
        
        model_names = []
        scores = []
        
        for model_name, results in self.comparison_results['models'].items():
            score = results['evaluation'].get(metric, None)
            if score is None and metric == 'throughput':
                score = results['performance']['throughput_analysis']['throughput_samples_per_s']
            elif score is None and metric == 'mean_latency_ms':
                score = results['performance']['latency_analysis']['mean_latency_ms']
            
            if score is not None:
                model_names.append(model_name)
                scores.append(score)
        
        if model_names and scores:
            bars = ax.bar(model_names, scores)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_ylabel(metric.replace("_", " ").title())
            
            # Rotate x-axis labels if needed
            if len(model_names) > 3:
                ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.2f}', ha='center', va='bottom') 