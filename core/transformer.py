import math
import os
import sys
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())  

from .network import MaskedMeanPooling

@torch.jit.export
def scaled_dot_product(q, k, v, mask: Optional [torch.Tensor]=None):
    '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
    '''
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_mask = mask.unsqueeze(1).unsqueeze(2)
        attn_logits = attn_logits.masked_fill(attn_mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 module number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask: Optional [torch.Tensor]=None):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        return o, attention

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """EncoderBlock.

        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.SiLU(inplace=True),      # 不会梯度消失
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional [torch.Tensor]=None):
        # Attention part
        attn_out, _ = self.attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(input_dim, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
    def forward(self, x, mask: Optional [torch.Tensor]=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)
    
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

class SingleChannelTransformer(nn.Module):
    def __init__(self, vocab_size, input_dim, model_dim, num_classes, num_heads, num_layers, dropout=0.0, input_dropout=0.0):
        super().__init__()
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Embedding(vocab_size, input_dim),
            nn.Dropout(input_dropout), 
            nn.Linear(input_dim, model_dim)
        )

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            input_dim = model_dim,
            dim_feedforward=2 * model_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        # Output classifier per sequence lement
        self.pooling_net = MaskedMeanPooling()
        # self.pooling_net = AttentionPooling(model_dim, model_dim)
        self.output_net = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            # nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask: Optional [torch.Tensor]=None, add_positional_encoding: bool=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)              # [Batch, SeqLen, ModDim]
        x = self.pooling_net(x, mask=mask)              # GlobalAveragePooling
        x = self.output_net(x)
        return x


class MultipleChannelTransformer(nn.Module):
    def __init__(self, vocab_size, input_dim, model_dim, num_classes, num_heads, num_layers, dropout=0.0, input_dropout=0.0):
        super().__init__()
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Embedding(vocab_size, input_dim),
            nn.Dropout(input_dropout), 
            nn.Linear(input_dim, model_dim)
        )

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            input_dim = model_dim,
            dim_feedforward=2 * model_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        # Output classifier per sequence lement
        self.pooling_net = MaskedMeanPooling()
        # self.pooling_net = AttentionPooling(model_dim, model_dim)
        self.output_net = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            # nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask: Optional [torch.Tensor]=None, add_positional_encoding: bool=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)              # [Batch, SeqLen, ModDim]
        x = self.pooling_net(x, mask=mask)              # GlobalAveragePooling
        x = self.output_net(x)
        return x
