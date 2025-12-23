import math
import torch.nn as nn
from model.DecoderLayer import DecoderLayer

class Decoder(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 dropout=0.1, pad_id=0, tie_weights=True):
        super().__init__()
        self.d_model = d_model
        
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.norm_final = nn.LayerNorm(d_model)

        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.out_proj.weight = self.embed.weight

    def forward(self, tgt_tokens, H_enc, self_mask=None, cross_mask=None):

        x = self.embed(tgt_tokens) * math.sqrt(self.d_model)
        
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, H_enc, self_mask=self_mask, cross_mask=cross_mask)

        x = self.norm_final(x)

        logits = self.out_proj(x)
        return logits