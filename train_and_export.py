import torch
from smart_transformer import SmartTransformer, AdaptiveConfig

# 1. Create a minimal config
config = AdaptiveConfig(
    vocab_size=1000,
    hidden_size=128,
    num_layers=2,
    num_attention_heads=4,
    intermediate_size=256,
    max_position_embeddings=64,
    use_adaptive_attention=True,
)

# 2. Initialize model
model = SmartTransformer(config)

# 3. Dummy training loop (just one step for demonstration)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
input_ids = torch.randint(0, config.vocab_size, (2, 10))
labels = input_ids.clone()

# Shift for language modeling: predict next token
shifted_input_ids = input_ids[:, :-1]
shifted_labels = labels[:, 1:]
outputs = model(input_ids=shifted_input_ids)
logits = outputs['logits']

# Align logits and shifted_labels
if logits.size(1) > shifted_labels.size(1):
    logits = logits[:, :shifted_labels.size(1), :]
elif shifted_labels.size(1) > logits.size(1):
    shifted_labels = shifted_labels[:, :logits.size(1)]

loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    shifted_labels.reshape(-1)
)
loss.backward()
optimizer.step()

# 4. Export the checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
}, 'smart_transformer_example.pt')

print("Checkpoint saved as smart_transformer_example.pt") 