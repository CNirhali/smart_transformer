import matplotlib.pyplot as plt
import numpy as np

# Example data (update with real HuggingFace results if available)
models = [
    'SmartTransformer',
    'PyTorch Transformer',
    'HuggingFace BERT',
    'HuggingFace GPT-2',
]
losses = [5.24, 7.09, 6.5, 7.0]  # Example/placeholder
speeds = [11.44, 4.21, 8.0, 10.0]  # ms/batch, example/placeholder
modularity = [5, 3, 4, 4]  # 1-5 scale
extensibility = [5, 3, 4, 4]  # 1-5 scale

x = np.arange(len(models))
width = 0.2

fig, ax1 = plt.subplots(figsize=(10, 5))
rects1 = ax1.bar(x - width, losses, width, label='Loss', color='tab:blue')
rects2 = ax1.bar(x, speeds, width, label='Time/Batch (ms)', color='tab:red')
ax1.set_ylabel('Loss / Time (ms)')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
rects3 = ax2.bar(x + width, modularity, width, label='Modularity', color='tab:green', alpha=0.5)
rects4 = ax2.bar(x + width*2, extensibility, width, label='Extensibility', color='tab:orange', alpha=0.5)
ax2.set_ylabel('Qualitative Score (1-5)')
ax2.set_ylim(0, 6)
ax2.legend(loc='upper right')

plt.title('Transformer Comparison: Loss, Speed, Modularity, Extensibility')
plt.tight_layout()
plt.savefig('transformer_comparison.png', dpi=200)
plt.close()
print("Comparison plot saved as 'transformer_comparison.png'.") 