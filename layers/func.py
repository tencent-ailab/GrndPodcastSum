import torch


def batch_index_select(x, idx):
    """
    :param x: *, n, Dim
    :param idx: *, k
    :return: *, k , Dim
    """
    idx_ = idx.unsqueeze(-1).expand(idx.shape + (x.shape[-1],))
    return torch.gather(x, -2, idx_)
