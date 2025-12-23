import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Tạo ma trận PE (max_len, d_model) cố định theo hàm Sin/Cos
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Thực hiện phép cộng vector vị trí vào embedding
        x = x + self.pe[:, :x.size(1), :]
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
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

    def forward(self, Q, K=None, V=None, mask=None):
        if K is None: K = Q
        if V is None: V = Q

        B, L_q, _ = Q.size()
        _, L_k, _ = K.size() # Lấy L_k từ K để tránh lỗi khi Q và K khác chiều dài (Cross-Attention)

        # 1. Chiếu và tách đầu (Split heads)
        q = self.W_Q(Q).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(K).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(V).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        # (B, H, L_q, D/H) * (B, H, D/H, L_k) -> (B, H, L_q, L_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            # Dùng giá trị cực âm để Softmax triệt tiêu các vị trí mask
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 3. Kết hợp với V và gộp đầu (Concatenate)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        
        return self.W_O(context)