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
        # Lưu ý: Đảm bảo trong lớp Encoder của bạn đã gọi lớp PositionalEncoding (dạng cộng)
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
        # Lưu ý: Đảm bảo trong lớp Decoder cũng dùng PositionalEncoding tương tự
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
        
        # Output Projection (Linear layer cuối cùng)
        self.out_proj = self.decoder.out_proj

    def make_src_mask(self, src):
        # Mask cho padding: (Batch, 1, 1, Seq_Len)
        mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)
        return mask

    def make_tgt_mask(self, tgt):
        B, L = tgt.shape
        # Padding mask cho target
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        
        # Causal mask (Tril) để không nhìn thấy tương lai
        # Theo bài báo gốc: "masking out values in the input of the softmax which correspond to illegal connections"
        causal_mask = torch.tril(torch.ones((L, L), device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        return pad_mask & causal_mask

    def forward(self, src, tgt):
        # 1. Tạo mask
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # 2. Encoder: Đầu vào sẽ được cộng Positional Encoding bên trong lớp Encoder
        enc_out = self.encoder(src, src_mask)
        
        # 3. Decoder: Sử dụng enc_out làm Memory cho Cross-Attention
        # tgt_mask (Self-attention) và src_mask (Cross-attention)
        logits = self.decoder(tgt, enc_out, self_mask=tgt_mask, cross_mask=src_mask)
        
        return logits