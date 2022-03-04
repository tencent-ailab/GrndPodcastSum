import math
from copy import deepcopy as cp

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


def clones(module, n):
    return nn.ModuleList([cp(module) for _ in range(n)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -6e4)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_k = config.d_model // config.n_head

        self.linears = clones(nn.Linear(self.d_model, self.d_model), 4)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        query, key, value = [l(x).reshape(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().reshape(batch_size, -1, self.n_head * self.d_k)
        return self.linears[3](x), attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(config.d_model, config.d_ff)
        self.w_2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super(SelfAttentionBlock, self).__init__()

        self.norm = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x, mask):
        x_ = self.norm(x)
        x_, attn = self.attn(x_, x_, x_, mask)
        return self.dropout(x_) + x, attn


class SourceAttentionBlock(nn.Module):
    def __init__(self, config):
        super(SourceAttentionBlock, self).__init__()

        self.norm = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x, m, mask):
        x_ = self.norm(x)
        x_, attn = self.attn(x_, m, m, mask)
        return self.dropout(x_) + x, attn


class FeedForwardBlock(nn.Module):
    def __init__(self, config):
        super(FeedForwardBlock, self).__init__()

        self.norm = nn.LayerNorm(config.d_model)
        self.feed_forward = PositionwiseFeedForward(config)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        x_ = self.norm(x)
        x_ = self.feed_forward(x_)
        return self.dropout(x_) + x


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.self_attn = SelfAttentionBlock(config)
        self.feed_forward = FeedForwardBlock(config)

    def forward(self, x, mask):
        x, attn = self.self_attn(x, mask)
        x = self.feed_forward(x)
        return x, attn


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()

        self.self_attn = SelfAttentionBlock(config)
        self.src_attn = SourceAttentionBlock(config)
        self.feed_forward = FeedForwardBlock(config)

    def forward(self, x, m, src_mask, tgt_mask):
        x, attn_tgt = self.self_attn(x, tgt_mask)
        x, attn_src = self.src_attn(x, m, src_mask)
        x = self.feed_forward(x)
        return x, attn_src, attn_tgt


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.layers = clones(EncoderBlock(config), config.n_layers)
        self.norms = clones(nn.LayerNorm(config.d_model), config.n_layers)

    def forward(self, x, mask):
        outputs = []
        attns = []
        for layer, norm in zip(self.layers, self.norms):
            x, attn = layer(x, mask)
            outputs.append(norm(x))
            attns.append(attn)
        return outputs, attns


class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super(PositionalEmbedding, self).__init__()

        p2e = torch.zeros(config.max_len, config.d_model)
        position = torch.arange(0.0, config.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, config.d_model, 2) * (- math.log(10000.0) / config.d_model))
        p2e[:, 0::2] = torch.sin(position * div_term)
        p2e[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('p2e', p2e)

    def forward(self, x):
        shp = x.size()
        emb = Variable(torch.index_select(self.p2e, 0, x.reshape(-1)), requires_grad=False).reshape(shp + (-1,))
        return emb


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.p2e = PositionalEmbedding(config)
        self.encoder = Encoder(config)

    def forward(self, input_emb, position_ids, attention_mask):
        # position embedding projection
        projection = self.p2e(position_ids) + input_emb
        return self.encoder(projection, attention_mask)
