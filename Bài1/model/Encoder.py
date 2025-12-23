import math
import torch.nn as nn
from model.EncoderLayer import EncoderLayer
# Import lớp đã có sẵn từ file khác (giả sử tên file là Attention.py)
from model.MultiHeadAttention import PositionalEncoding 

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, 
                 dropout=0.1, pad_id=0):
        super().__init__()
        self.d_model = d_model
        
        # 1. Khởi tạo Embedding
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        
        # 2. Khởi tạo Positional Encoding (Phép cộng)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 3. Danh sách các lớp Encoder Layer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, src_tokens, src_mask=None):
        # Bước 1: Word Embedding & Scaling (nhân căn d_model)
        x = self.embed(src_tokens) * math.sqrt(self.d_model)
        
        # Bước 2: CỘNG Positional Encoding (Thay thế cho RoPE)
        x = self.pos_encoding(x)
        
        # Bước 3: Dropout sau khi cộng vị trí
        x = self.dropout(x)

        # Bước 4: Chạy qua các lớp Encoder Layers
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)

        return self.norm_final(x)