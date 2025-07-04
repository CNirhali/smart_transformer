"""
Adaptive Attention Mechanisms

This module implements various attention mechanisms that adapt to different
input characteristics and computational requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
from einops import rearrange, repeat
import warnings

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    warnings.warn("Flash attention not available. Using standard attention.")

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers not available. Using standard attention.")


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention mechanism that automatically selects the best attention
    strategy based on input characteristics and available hardware.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Attention projections
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Attention type selection
        self.use_flash_attention = config.use_flash_attention and FLASH_ATTN_AVAILABLE
        self.use_xformers = XFORMERS_AVAILABLE
        
        # Adaptive parameters
        self.attention_threshold = 512  # Switch to flash attention for longer sequences
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        
        # Ensure hidden_states is 3D: (batch_size, seq_length, hidden_size)
        if hidden_states.dim() > 3:
            hidden_states = hidden_states.view(hidden_states.size(0), -1, hidden_states.size(-1))
        assert hidden_states.dim() == 3, f"hidden_states must be 3D, got {hidden_states.shape}"
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Project queries, keys, and values
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key/value states
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        # Choose attention mechanism based on sequence length and available optimizations
        if seq_length > self.attention_threshold and self.use_flash_attention:
            attention_output = self._flash_attention(
                query_states, key_states, value_states, attention_mask
            )
        elif self.use_xformers:
            attention_output = self._xformers_attention(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attention_output = self._standard_attention(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        
        # Project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(attention_output.size(0), attention_output.size(1), hidden_size)
        attention_output = self.output(attention_output)
        
        outputs = (attention_output,)
        
        if use_cache:
            outputs = outputs + ((key_states, value_states),)
        else:
            outputs = outputs + (None,)
        
        if output_attentions:
            outputs = outputs + (None,)  # Attention weights not computed for optimized attention
        
        return outputs
    
    def _flash_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Use Flash Attention for efficient computation."""
        
        # Flash attention expects (batch_size, seq_len, num_heads, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Apply scaling
        query_states = query_states * self.scaling
        
        # Flash attention
        attention_output = flash_attn_func(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            softmax_scale=None,
            causal=True
        )
        
        return attention_output.transpose(1, 2)
    
    def _xformers_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Use xFormers for efficient attention computation."""
        
        # xFormers attention
        attention_output = xops.memory_efficient_attention(
            query_states, key_states, value_states,
            attn_bias=attention_mask,
            op=None
        )
        
        return attention_output
    
    def _standard_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        
        batch_size, num_heads, query_seq_length, _ = query_states.size()
        _, _, key_seq_length, _ = key_states.size()
        
        # Compute attention scores
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores * self.scaling
        
        # Create causal mask (lower triangular) - use max sequence length
        max_seq_length = max(query_seq_length, key_seq_length)
        causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, device=attention_scores.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, max_seq_length, max_seq_length)
        causal_mask = causal_mask[:1, :1, :query_seq_length, :key_seq_length]  # Crop to actual sizes
        causal_mask = (1.0 - causal_mask) * -1e9  # Mask future positions with large negative value
        
        # Apply attention mask (padding mask)
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_length) - should match key_seq_length
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]  # (batch_size, 1, 1, seq_length)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, :, None, :]  # (batch_size, 1, 1, seq_length)
            
            # Ensure attention_mask matches key_seq_length
            current_mask_length = attention_mask.size(-1)
            if current_mask_length < key_seq_length:
                # Pad the mask with ones (valid tokens) for the additional positions
                padding = torch.ones(attention_mask.size(0), 1, 1, key_seq_length - current_mask_length, 
                                   device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, padding], dim=-1)
            elif current_mask_length > key_seq_length:
                # Truncate the mask to match key_seq_length
                attention_mask = attention_mask[:, :, :, :key_seq_length]
            
            # Broadcast to (batch_size, 1, query_seq_length, key_seq_length)
            attention_mask = attention_mask.expand(attention_mask.size(0), 1, query_seq_length, key_seq_length)
            # Combine with causal mask
            combined_mask = causal_mask + (1.0 - attention_mask) * -1e9
        else:
            combined_mask = causal_mask
        
        attention_scores = attention_scores + combined_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply head mask
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # Compute attention output
        attention_output = torch.matmul(attention_probs, value_states)
        
        return attention_output


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism that processes information at different
    temporal and spatial scales simultaneously.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        # Multiple attention heads for different scales
        self.scales = [1, 2, 4, 8]  # Different attention scales
        self.scale_heads = nn.ModuleDict({
            f'scale_{scale}': nn.MultiheadAttention(
                self.hidden_size, 
                self.num_heads // len(self.scales),
                dropout=config.attention_dropout,
                batch_first=True
            ) for scale in self.scales
        })
        
        # Scale fusion
        self.scale_fusion = nn.Linear(self.hidden_size * len(self.scales), self.hidden_size)
        
        # Output projection
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Process each scale
        scale_outputs = []
        for scale in self.scales:
            # Create scale-specific attention mask
            if attention_mask is not None:
                scale_mask = self._create_scale_mask(attention_mask, scale)
            else:
                scale_mask = None
            
            # Apply attention at this scale
            scale_output, _ = self.scale_heads[f'scale_{scale}'](
                hidden_states, hidden_states, hidden_states,
                attn_mask=scale_mask,
                need_weights=output_attentions
            )
            
            scale_outputs.append(scale_output)
        
        # Concatenate scale outputs
        multi_scale_output = torch.cat(scale_outputs, dim=-1)
        
        # Fuse scales
        fused_output = self.scale_fusion(multi_scale_output)
        
        # Final output projection
        attention_output = self.output(fused_output)
        
        outputs = (attention_output,)
        
        if use_cache:
            outputs = outputs + (None,)  # No key/value caching for multi-scale
        else:
            outputs = outputs + (None,)
        
        if output_attentions:
            outputs = outputs + (None,)  # Attention weights not computed for multi-scale
        
        return outputs
    
    def _create_scale_mask(self, attention_mask: torch.Tensor, scale: int) -> torch.Tensor:
        """Create attention mask for a specific scale."""
        # This is a simplified implementation
        # In practice, you might want more sophisticated scale-specific masking
        return attention_mask


class SparseAttention(nn.Module):
    """
    Sparse attention mechanism that only attends to a subset of positions,
    reducing computational complexity.
    """
    
    def __init__(self, config, sparsity_factor: int = 4):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.sparsity_factor = sparsity_factor
        
        # Attention projections
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Project queries, keys, and values
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Create sparse attention pattern
        sparse_indices = self._get_sparse_indices(seq_length)
        
        # Apply sparse attention
        attention_output = self._sparse_attention(
            query_states, key_states, value_states, sparse_indices, attention_mask
        )
        
        # Project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_length, hidden_size)
        attention_output = self.output(attention_output)
        
        outputs = (attention_output,)
        
        if use_cache:
            outputs = outputs + ((key_states, value_states),)
        else:
            outputs = outputs + (None,)
        
        if output_attentions:
            outputs = outputs + (None,)  # Attention weights not computed for sparse attention
        
        return outputs
    
    def _get_sparse_indices(self, seq_length: int) -> torch.Tensor:
        """Get indices for sparse attention pattern."""
        # Local attention pattern
        indices = []
        for i in range(seq_length):
            # Attend to local window
            start = max(0, i - self.sparsity_factor)
            end = min(seq_length, i + self.sparsity_factor + 1)
            indices.extend([(i, j) for j in range(start, end)])
        
        return torch.tensor(indices, dtype=torch.long)
    
    def _sparse_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        sparse_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply sparse attention using the given indices."""
        
        batch_size, num_heads, seq_length, head_dim = query_states.size()
        
        # Extract sparse attention scores
        query_sparse = query_states[:, :, sparse_indices[:, 0], :]
        key_sparse = key_states[:, :, sparse_indices[:, 1], :]
        
        # Compute attention scores
        attention_scores = torch.matmul(query_sparse, key_sparse.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(head_dim)
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply to values
        value_sparse = value_states[:, :, sparse_indices[:, 1], :]
        attention_output = torch.matmul(attention_probs, value_sparse)
        
        # Reconstruct full sequence (simplified)
        # In practice, you'd need more sophisticated reconstruction
        attention_output = attention_output.view(batch_size, num_heads, seq_length, head_dim)
        
        return attention_output


class LinearAttention(nn.Module):
    """
    Linear attention mechanism that reduces computational complexity from O(n²) to O(n).
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        # Attention projections
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Feature map for linear attention
        self.feature_map = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 2),
            nn.ReLU(),
            nn.Linear(self.head_dim * 2, self.head_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Project queries, keys, and values
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply feature map
        query_states = self.feature_map(query_states)
        key_states = self.feature_map(key_states)
        
        # Linear attention computation
        attention_output = self._linear_attention(query_states, key_states, value_states)
        
        # Project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(attention_output.size(0), attention_output.size(1), hidden_size)
        attention_output = self.output(attention_output)
        
        outputs = (attention_output,)
        
        if use_cache:
            outputs = outputs + ((key_states, value_states),)
        else:
            outputs = outputs + (None,)
        
        if output_attentions:
            outputs = outputs + (None,)  # Attention weights not computed for linear attention
        
        return outputs
    
    def _linear_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute linear attention."""
        
        # Compute KV product
        kv = torch.matmul(key_states.transpose(-1, -2), value_states)
        
        # Compute normalization factor
        k_sum = key_states.sum(dim=-2, keepdim=True)
        
        # Compute attention output
        attention_output = torch.matmul(query_states, kv) / (torch.matmul(query_states, k_sum.transpose(-1, -2)) + 1e-8)
        
        return attention_output 