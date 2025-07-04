import matplotlib.pyplot as plt
import numpy as np

models = [
    'SmartTransformer',
    'PyTorch Transformer',
    'HF BERT',
    'HF GPT-2',
    'Longformer',
    'Performer',
]
losses = [5.24, 7.09, 6.5, 7.0, 7.0, 7.0]
speeds = [11.44, 4.21, 8.0, 10.0, 12.0, 9.0]
modularity = [5, 3, 4, 4, 3, 3]
extensibility = [5, 3, 4, 4, 3, 3]
max_seq = [2048, 1024, 512, 1024, 4096, 4096]

x = np.arange(len(models))
width = 0.15

fig, ax1 = plt.subplots(figsize=(12, 6))
rects1 = ax1.bar(x - 2*width, losses, width, label='Loss', color='tab:blue')
rects2 = ax1.bar(x - width, speeds, width, label='Time/Batch (ms)', color='tab:red')
rects3 = ax1.bar(x, modularity, width, label='Modularity (1-5)', color='tab:green', alpha=0.7)
rects4 = ax1.bar(x + width, extensibility, width, label='Extensibility (1-5)', color='tab:orange', alpha=0.7)
rects5 = ax1.bar(x + 2*width, max_seq, width, label='Max Seq Len', color='tab:purple', alpha=0.5)

ax1.set_ylabel('Metric Value')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15)
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Detailed Transformer Comparison')
plt.tight_layout()
plt.savefig('detailed_transformer_comparison.png', dpi=200)
plt.close()
print("Detailed comparison plot saved as 'detailed_transformer_comparison.png'.") 