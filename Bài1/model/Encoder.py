import math
import torch.nn as nn
from model.EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, 
                 dropout=0.1, pad_id=0):
        super().__init__()
        self.d_model = d_model
        
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, src_tokens, src_mask=None):
        x = self.embed(src_tokens) * math.sqrt(self.d_model)
        
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask=src_mask)

        return self.norm_final(x)