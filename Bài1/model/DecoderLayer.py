import torch.nn as nn
from model.MultiHeadAttention import MultiHeadAttention
from model.FeedForward import FeedForward
from model.NormAndAdd import NormAndAdd

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Các lớp Attention bây giờ là chuẩn (Scaled Dot-Product)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        
        self.norm_add_1 = NormAndAdd(d_model, dropout) 
        self.norm_add_2 = NormAndAdd(d_model, dropout)
        self.norm_add_3 = NormAndAdd(d_model, dropout)

    def forward(self, x, H_enc, self_mask=None, cross_mask=None):
        # 1. Masked Self-Attention
        # Loại bỏ use_rope=True
        x = self.norm_add_1(x, lambda u: self.self_attn(u, mask=self_mask))

        # 2. Cross-Attention (Kết nối với Encoder)
        # Q đến từ Decoder (u), K và V đến từ Encoder (H_enc)
        # Loại bỏ use_rope=False
        x = self.norm_add_2(x, lambda u: self.cross_attn(Q=u, K=H_enc, V=H_enc, mask=cross_mask))

        # 3. Feed Forward
        x = self.norm_add_3(x, lambda u: self.ffn(u))

        return x