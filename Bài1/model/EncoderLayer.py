import torch.nn as nn
from model.MultiHeadAttention import MultiHeadAttention
from model.FeedForward import FeedForward
from model.NormAndAdd import NormAndAdd 

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        
        self.norm_add_1 = NormAndAdd(d_model, dropout)
        self.norm_add_2 = NormAndAdd(d_model, dropout)

    def forward(self, x, src_mask=None):
       
        x = self.norm_add_1(x, lambda u: self.self_attn(u, mask=src_mask, use_rope=True))

        x = self.norm_add_2(x, lambda u: self.ffn(u))

        return x