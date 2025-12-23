import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()


    def forward(self, x):
        h = self.linear1(x)
        h = self.act(h)
        h = self.dropout(h)
        out = self.linear2(h)
        return out