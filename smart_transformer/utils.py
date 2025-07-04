import torch

def align_logits_and_labels(logits, labels):
    """Align sequence length of logits and labels for loss computation."""
    if logits.size(1) > labels.size(1):
        logits = logits[:, :labels.size(1), :]
    elif labels.size(1) > logits.size(1):
        labels = labels[:, :logits.size(1)]
    return logits, labels

def filter_batch_for_model(batch, allowed_keys=("input_ids", "attention_mask", "token_type_ids")):
    """Filter a batch dictionary to only include keys allowed by the model."""
    if isinstance(batch, dict):
        return {k: v for k, v in batch.items() if k in allowed_keys}
    return batch 