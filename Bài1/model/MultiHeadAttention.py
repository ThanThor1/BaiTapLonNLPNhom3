import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x_rot

def _precompute_rope(dim, max_len=500, theta=10000.0, device=None, dtype=None):
    if dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {dim}")

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)

    if dtype is not None:
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)

    return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (_rotate_half(x) * sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, rope_max_len=5000, rope_theta=10000.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        cos, sin = _precompute_rope(self.head_dim, max_len=rope_max_len, theta=rope_theta)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _get_rope(self, seq_len, device, dtype):
        if seq_len <= self.cos_cached.size(0):
            cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
            sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
            return cos, sin

        cos, sin = _precompute_rope(self.head_dim, max_len=seq_len, device=device, dtype=dtype)
        return cos, sin

    def forward(self, Q, K=None, V=None, mask=None, use_rope=True):
        if K is None:
            K = Q
        if V is None:
            V = Q

        B, L_q, _ = Q.size()
        _, L_k, _ = K.size()

        q = self.W_Q(Q).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(K).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(V).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        if use_rope:
            cos_q, sin_q = self._get_rope(L_q, device=q.device, dtype=q.dtype)
            cos_q = cos_q.view(1, 1, L_q, self.head_dim)
            sin_q = sin_q.view(1, 1, L_q, self.head_dim)
            q = apply_rotary_pos_emb(q, cos_q, sin_q)

            cos_k, sin_k = self._get_rope(L_k, device=k.device, dtype=k.dtype)
            cos_k = cos_k.view(1, 1, L_k, self.head_dim)
            sin_k = sin_k.view(1, 1, L_k, self.head_dim)
            k = apply_rotary_pos_emb(k, cos_k, sin_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        output = self.W_O(context)
        return output
