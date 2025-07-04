import torch
import time
from smart_transformer import SmartTransformer, AdaptiveConfig
from torch.nn import Transformer as TorchTransformer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from smart_transformer.utils import align_logits_and_labels
import matplotlib.pyplot as plt
import json

# Simple synthetic dataset
vocab_size = 1000
num_samples = 200
seq_length = 32
batch_size = 4
input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
labels = input_ids.clone()
dataset = TensorDataset(input_ids, labels)
dataloader = DataLoader(dataset, batch_size=batch_size)

# SmartTransformer config
smart_config = AdaptiveConfig(
    vocab_size=vocab_size,
    hidden_size=128,
    num_layers=2,
    num_attention_heads=4,
    intermediate_size=256,
    max_position_embeddings=64,
    use_adaptive_attention=True,
)
smart_model = SmartTransformer(smart_config)

# Baseline: PyTorch nn.Transformer
baseline_model = TorchTransformer(
    d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, batch_first=True
)
baseline_embedding = torch.nn.Embedding(vocab_size, 128)
baseline_output_proj = torch.nn.Linear(128, vocab_size)

# Benchmark function
def benchmark_model(model, name):
    model.eval()
    total_time = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for input_ids, labels in dataloader:
            start = time.time()
            if isinstance(model, SmartTransformer):
                outputs = model(input_ids=input_ids)
                logits = outputs['logits']
            else:
                src = baseline_embedding(input_ids)
                tgt = baseline_embedding(input_ids)
                baseline_out = model(src, tgt)  # [batch, seq, d_model]
                logits = baseline_output_proj(baseline_out)  # [batch, seq, vocab_size]
            logits, labels = align_logits_and_labels(logits, labels)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item()
            total_time += time.time() - start
    avg_loss = total_loss / len(dataloader)
    avg_time = total_time / len(dataloader)
    print(f"{name}: Avg Loss={avg_loss:.4f}, Avg Time/Batch={avg_time*1000:.2f} ms")
    return avg_loss, avg_time

print("\n=== Benchmarking SmartTransformer ===")
smart_loss, smart_time = benchmark_model(smart_model, "SmartTransformer")

print("\n=== Benchmarking PyTorch nn.Transformer ===")
baseline_loss, baseline_time = benchmark_model(baseline_model, "PyTorch Transformer")

print("\n=== Benchmark Summary ===")
print(f"SmartTransformer: Loss={smart_loss:.4f}, Time/Batch={smart_time*1000:.2f} ms")
print(f"PyTorch Transformer: Loss={baseline_loss:.4f}, Time/Batch={baseline_time*1000:.2f} ms")

# After printing summary
results_dict = {
    'SmartTransformer': {'loss': smart_loss, 'time_per_batch_ms': smart_time*1000},
    'PyTorch Transformer': {'loss': baseline_loss, 'time_per_batch_ms': baseline_time*1000},
}
with open('benchmark_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

# Bar plot
labels = list(results_dict.keys())
losses = [results_dict[k]['loss'] for k in labels]
times = [results_dict[k]['time_per_batch_ms'] for k in labels]

fig, ax1 = plt.subplots(figsize=(7, 4))
color = 'tab:blue'
ax1.set_xlabel('Model')
ax1.set_ylabel('Loss', color=color)
ax1.bar(labels, losses, color=color, alpha=0.7, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Time/Batch (ms)', color=color)
ax2.plot(labels, times, color=color, marker='o', label='Time/Batch')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('SmartTransformer vs. PyTorch Transformer\n(Loss and Time per Batch)')
plt.savefig('benchmark_plot.png', dpi=200)
plt.close()
print("\nBenchmark results saved as 'benchmark_results.json' and 'benchmark_plot.png'.") 