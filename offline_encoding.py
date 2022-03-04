import time

import numpy as np
import torch

from layers.extractor import TokenLevelEncoder
from test import get_data
from utility import DataParallelModel, save_to_pkl
from utility.prepare_data import prepare_offline_data


def weights(c=0.8, k=64):
    r = c / k
    ret = []
    for i in range(k):
        w = 1 - i * r
        ret.append(w)
    return ret


parts = list(range(5, 100, 5))


def prepare_data_cuda(batch_data, config, func=None):
    ret = func(batch_data, config)
    cuda_ret = ()
    for it in list(ret):
        if type(it) != list:
            cuda_ret += (it.cuda(config.device),)
        else:
            cuda_ret += (it, )
    return cuda_ret


def offline(config, log):
    __, valid_set = get_data(config, log, mode="test")

    net = TokenLevelEncoder(config).cuda(config.device)
    if config.parallel:
        net = DataParallelModel(net)

    h = {}
    st = time.time()
    for batch_idx, batch_data in enumerate(valid_set):
        batch_chunks, batch_indices, batch_lengths, __ = \
            prepare_data_cuda(batch_data, config, prepare_offline_data)
        bsz, n, l = batch_chunks.shape
        k = config.offline_k // torch.cuda.device_count() // bsz
        m = n // k
        hiddens_np = np.empty((bsz, 0, config.doc_encoder.d_model), dtype=np.float64)
        for i in range(m):
            batch_chunks_ = batch_chunks[:, i * k: (i + 1) * k]
            with torch.no_grad():
                hiddens = net(batch_chunks_)
            if config.parallel:
                hiddens = torch.cat([it.to("cuda:0") for it in hiddens], dim=0)
            hiddens_np = np.concatenate([hiddens_np, hiddens.cpu().numpy()], axis=1)

        if m * k < n:
            batch_chunks_ = batch_chunks[:, m * k: n]
            with torch.no_grad():
                hiddens = net(batch_chunks_)
            if config.parallel:
                hiddens = torch.cat([it.to("cuda:0") for it in hiddens], dim=0)
            hiddens_np = np.concatenate([hiddens_np, hiddens.cpu().numpy()], axis=1)

        for i, (idx, l) in enumerate(zip(batch_indices, batch_lengths)):
            h[idx] = hiddens_np[i][:l, :]

        print(batch_idx + 1, hiddens_np.shape, batch_indices, (time.time() - st))
    name = "hidden_v_"+str(config.kernel_size)+"_"+str(config.stride)+"_"+config.window_type+".pkl"
    save_to_pkl(name, h)
