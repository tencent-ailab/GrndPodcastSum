import torch
import torch.nn as nn


class BinaryClsHead(nn.Module):
    def __init__(self, n_in, n_out, dropout):
        super().__init__()
        self.dense = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(n_out, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x.squeeze(-1)


class MultiClsHead(nn.Module):
    def __init__(self, n_in, n_inner, n_out, dropout):
        super().__init__()
        self.dense = nn.Linear(n_in, n_inner)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(n_inner, n_out)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
