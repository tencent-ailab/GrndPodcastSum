import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .func import batch_index_select


class LabelSmoothing(nn.Module):
    def __init__(self, config):
        super(LabelSmoothing, self).__init__()
        self.crit = nn.KLDivLoss(size_average=False)
        self.pad_idx = config.pad
        self.confidence = 1.0 - config.label_smoothing
        self.smoothing = config.label_smoothing
        self.size = config.n_vocab

    def forward(self, predicts, target):
        assert self.size == predicts.size(1)
        dist = torch.full_like(predicts, self.smoothing / (self.size - 2))
        dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        dist[:, self.pad_idx] = 0
        mask_idx = torch.nonzero(target.data == self.pad_idx)
        if mask_idx.dim() > 0:
            dist.index_fill_(0, mask_idx.squeeze(), 0.0)
        return self.crit(predicts, Variable(dist, requires_grad=False))


class KLDivLoss(nn.Module):
    def __init__(self, config):
        super(KLDivLoss, self).__init__()
        self.crit = LabelSmoothing(config)

    def forward(self, predicts, target, norm=1.0):
        loss = self.crit(predicts.contiguous().view(-1, predicts.size(-1)), target.contiguous().view(-1))
        return loss / norm


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.crit = nn.BCELoss(reduction='sum')

    def forward(self, input, target, norm= 1.0):
        return self.crit(input, target) / norm


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def forward(self, inputs, target, weights=None, norm=1.0):
        inputs = inputs.reshape(-1)
        target = target.reshape(-1)
        weights = weights.reshape(-1)
        return F.binary_cross_entropy_with_logits(inputs, target, weights, reduction="sum") / norm


class CrossEntropy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ignore_index = config.ignore_index
        self.crit = nn.CrossEntropyLoss(ignore_index=config.ignore_index, reduction="sum")

    def forward(self, pred, target):
        target = target.reshape(-1)
        num = target.shape[0]
        pred = pred.reshape(num, -1)
        return self.crit(pred, target)


class AdjacentRegularizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ignore_index = config.ignore_index

    def forward(self, pred, tgt_indices, tgt_mask, src_mask):
        """
            pred: bsz, tgt_len, src_len
            target_indices: bsz, n
            tgt_mask: bsze, n
            src_mask: bsz, src_len
        """
        if tgt_indices.shape[1] < 2:
            return 0
        pred_ = batch_index_select(pred, tgt_indices)
        pred_ += (src_mask.unsqueeze(1) - 1) * 1e6
        prob = torch.softmax(pred_, dim=-1)
        prob_sum = torch.cumsum(prob, dim=-1)
        diff = prob_sum[:, 1:, :] - prob_sum[:, :-1, :]
        #print(diff.shape, tgt_mask.shape, src_mask.shape)
        #diff = torch.where(diff > 0, diff, torch.zeros_like(diff))
        loss = torch.relu(diff) * tgt_mask.unsqueeze(2) * src_mask.unsqueeze(1)
        return loss.sum()


class CrossEntropyWithRegularizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.coef = config.coef
        self.crit = CrossEntropy(config)
        if config.regularizer == 0:
            self.regularizer = None
        elif config.regularizer == 1:
            self.regularizer = AdjacentRegularizer(config)

    def regu(self, *args):
        if self.regularizer is None:
            return 0
        return self.regularizer(*args)

    def loss(self, *args):
        return self.crit(*args)

    def forward(self, pred, target, tgt_indices, tgt_mask, src_mask, norm_loss=1.0, norm_regu=1.0):
        loss = self.crit(pred, target) / norm_loss
        regu = self.regu(pred, tgt_indices, tgt_mask, src_mask) / norm_regu

        if loss < 0:
            print("Negative Loss Error", norm_loss)

        if regu < 0:
            print("Negative Regu Error", norm_regu)
        return loss + self.coef * regu
