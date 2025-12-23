import torch
import torch.nn as nn
from model.Encoder import Encoder
from model.Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, 
                 num_encoder_layers, num_decoder_layers, dropout, pad_id, tie_weights=True):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.pad_id = pad_id
        
        # Encoder 
        self.encoder = Encoder(
            vocab_size=src_vocab_size, 
            d_model=d_model, 
            num_heads=num_heads, 
            d_ff=d_ff, 
            num_layers=num_encoder_layers, 
            dropout=dropout,
            pad_id=pad_id
        )
        
        # Decoder
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size, 
            d_model=d_model, 
            num_heads=num_heads, 
            d_ff=d_ff, 
            num_layers=num_decoder_layers, 
            dropout=dropout,
            pad_id=pad_id,
            tie_weights=tie_weights 
        )
        
        # Output Projection
        self.out_proj = self.decoder.out_proj

    def make_src_mask(self, src):
        mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)
        return mask

    def make_tgt_mask(self, tgt):
        B, L = tgt.shape

        k_pad = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)   
        q_pad = (tgt != self.pad_id).unsqueeze(1).unsqueeze(3)  

        causal = torch.tril(torch.ones((L, L), device=tgt.device, dtype=torch.bool))  
        causal = causal.unsqueeze(0).unsqueeze(0)  

        return k_pad & q_pad & causal


    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        enc_out = self.encoder(src, src_mask)
        
        logits = self.decoder(tgt, enc_out, self_mask=tgt_mask, cross_mask=src_mask)
        
        return logits