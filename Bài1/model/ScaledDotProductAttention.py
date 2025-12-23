import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_head = q.size(-1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        
        return context, attn_weights