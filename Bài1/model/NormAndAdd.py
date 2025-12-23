import torch.nn as nn

class NormAndAdd(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        norm_x = self.norm(x)
        output = sublayer(norm_x)
        
        return x + self.dropout(output)