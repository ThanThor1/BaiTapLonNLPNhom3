import math
import torch.nn as nn
from model.DecoderLayer import DecoderLayer
# Import lớp PositionalEncoding từ file bạn đã định nghĩa (ví dụ model/Attention.py)
from model.MultiHeadAttention import PositionalEncoding 

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 dropout=0.1, pad_id=0, tie_weights=True):
        super().__init__()
        self.d_model = d_model
        
        # 1. Word Embedding cho Target
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # 2. Khởi tạo Positional Encoding (Phép cộng)
        self.pos_encoding = PositionalEncoding(d_model)

        # 3. Danh sách các lớp Decoder Layer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm_final = nn.LayerNorm(d_model)

        # 4. Đầu ra Projection
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Kỹ thuật chia sẻ trọng số (Weight Tying)
        if tie_weights:
            self.out_proj.weight = self.embed.weight

    def forward(self, tgt_tokens, H_enc, self_mask=None, cross_mask=None):
        # Bước 1: Embedding & Scaling (chuẩn bài báo gốc)
        x = self.embed(tgt_tokens) * math.sqrt(self.d_model)
        
        # Bước 2: CỘNG Positional Encoding (Thay thế RoPE)
        x = self.pos_encoding(x)
        
        # Bước 3: Dropout sau khi cộng vị trí
        x = self.dropout(x)

        # Bước 4: Chạy qua các lớp Decoder Layers
        for layer in self.layers:
            # Lưu ý: Mỗi layer thực hiện cả Self-Attention và Cross-Attention với H_enc
            x = layer(x, H_enc, self_mask=self_mask, cross_mask=cross_mask)

        x = self.norm_final(x)

        # Bước 5: Tạo Logits
        logits = self.out_proj(x)
        return logits