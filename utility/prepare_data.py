import re
from copy import deepcopy as cp

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BartTokenizer

from .generation_tricks import get_n_gram

Tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
translator = re.compile('[%s]' % re.escape('!#$%&()*+,./:;<=>?@[]^_{|}~'))
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
              'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
              'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
              'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
              'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
              "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
              "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
              'wouldn', "wouldn't"]
stop_bigrams = set([(a, b) for a in stop_words for b in stop_words])


def splitter(x):
    return (translator.sub(' ', x)).split()


def adding_cls_sep_2d(inputs, cls, sep):
    new_inputs = []
    for seq in inputs:
        new_seq = [cls] + seq + [sep]
        new_inputs.append(new_seq)
    return new_inputs


def adding_cls_sep_3d(inputs, cls, sep):
    new_inputs = []
    for inst in inputs:
        new_inst = adding_cls_sep_2d(inst, cls, sep)
        new_inputs.append(new_inst)
    return new_inputs


def pad_2d_inputs(inputs, pad, t=torch.LongTensor):
    lengths = [len(it) for it in inputs]
    max_len = max(lengths)
    padded_inputs = []
    for seq in inputs:
        new_seq = t(seq)
        padded_seq = F.pad(new_seq, (0, max_len - len(seq)), "constant", pad)
        padded_inputs.append(padded_seq)
    padded_inputs = torch.stack(padded_inputs)
    return padded_inputs


def pad_3d_inputs(inputs, pad):
    """
    :param inputs: Batch, Number, Length
    :param pad: int
    :return: padded_torch_tensor
    """
    numbers = [len(i) for i in inputs]
    lengths = [len(j) for i in inputs for j in i]
    max_num = max(numbers)
    max_len = max(lengths)

    padded_inputs = []
    for inst in inputs:
        padded_inst = []
        for seq in inst:
            seq_j = torch.LongTensor(seq)
            padded_seq = F.pad(seq_j, (0, max_len - len(seq)), "constant", pad)
            padded_inst.append(padded_seq)
        padded_inst = torch.stack(padded_inst)
        padded_inst = F.pad(padded_inst, (0, 0, 0, max_num - len(inst)), "constant", pad)
        padded_inputs.append(padded_inst)
    padded_inputs = torch.stack(padded_inputs)
    return padded_inputs


def pad_4d_inputs(inputs, pad):
    """
    :param inputs: Batch, D1, D2, Length
    :param pad: int
    :return: padded_torch_tensor
    """
    d_3 = [len(i) for i in inputs]
    d_2 = [len(j) for i in inputs for j in i]
    d_1 = [len(k) for i in inputs for j in i for k in j]

    max_d1 = max(d_1)
    max_d2 = max(d_2)
    max_d3 = max(d_3)

    padded_inputs = []
    for t3 in inputs:
        padded_i = []
        for t2 in t3:
            padded_j = []
            for t1 in t2:
                t1 = torch.LongTensor(t1)
                padded_k = F.pad(t1, (0, max_d1 - len(t1)), "constant", pad)
                padded_j.append(padded_k)
            padded_j = torch.stack(padded_j)
            padded_j = F.pad(padded_j, (0, 0, 0, max_d2 - len(t2)), "constant", pad)
            padded_i.append(padded_j)
        padded_i = torch.stack(padded_i)
        padded_i = F.pad(padded_i, (0, 0, 0, 0, 0, max_d3 - len(t3)), "constant", pad)
        padded_inputs.append(padded_i)
    padded_inputs = torch.stack(padded_inputs)
    return padded_inputs


def merge2d(data_1, data_2):
    new_data = []
    for inst_1, inst_2 in zip(data_1, data_2):
        new_data.append(cp(inst_1) + cp(inst_2))
    return new_data


def cross_merge2d(data_1, data_2):
    ret = []
    for inst_1 in data_1:
        ret_i = []
        for inst_2 in data_2:
            ret_i.append(cp(inst_2) + cp(inst_1))
        ret.append(ret_i)
    return ret


def cross_merge3d(data_1, data_2):
    ret = []
    for inst_1, inst_2 in zip(data_1, data_2):
        ret.append(cross_merge2d(inst_1, inst_2))
    return ret


def plain_text(inputs, head=None):
    if head is None:
        return [token for sent in inputs for token in sent]
    else:
        return [token for sent in inputs for token in [head] + sent]


def create_chunks_for_instance_token(doc_inputs, kernel_size, stride, output=False):
    plain_doc_inputs = plain_text(doc_inputs)
    n = len(doc_inputs)
    len_sent = [len(it) for it in doc_inputs]
    c = np.asarray([0] + len_sent).cumsum()
    l = c[-1]
    st = list(range(0, l, stride))
    ed = [min(it + kernel_size, l) for it in st]
    new_st = []
    total_sents = 0
    i = 0
    for it in st:
        while c[i] < it:
            i += 1
        new_st.append(c[i])
        total_sents -= i
    new_ed = []
    j = 0
    for it in ed:
        while (j < n) and c[j + 1] <= it:
            j += 1
        new_ed.append(c[j])
        total_sents += j
    chunks = []
    for start, end in zip(new_st, new_ed):
        new_chunk = cp(plain_doc_inputs[start:end])
        chunks.append(new_chunk)
    if output:
        return chunks, total_sents
    return chunks


def create_chunks_for_instance_sent(doc_inputs, kernel_size, stride, output=False):
    n = len(doc_inputs)
    st = list(range(0, n, stride))
    ed = [min(it + kernel_size, n) for it in st]
    chunks = []
    total_sents = 0
    for start, end in zip(st, ed):
        chunk = cp(plain_text(doc_inputs[start: end]))
        k = end
        while len(chunk) > 510:
            k -= 1
            chunk = cp(plain_text(doc_inputs[start: k]))
        chunks.append(chunk)
        total_sents += k - start
    if output:
        return chunks, total_sents
    return chunks


def create_chunks_for_batch(batch_docs, config):
    batch_chunks = []
    for doc_inputs in batch_docs:
        if config.window_type == "token":
            batch_chunks.append(create_chunks_for_instance_token(doc_inputs, config.kernel_size, config.stride))
        elif config.window_type == "sent":
            batch_chunks.append(create_chunks_for_instance_sent(doc_inputs, config.kernel_size, config.stride))
    return batch_chunks


def create_chunks_4_batch_w_total_sents(batch_docs, config):
    batch_chunks = []
    total_sents = 0
    for doc_inputs in batch_docs:
        if config.window_type == "token":
            chunks, n_sents = create_chunks_for_instance_token(doc_inputs, config.kernel_size, config.stride, True)
            batch_chunks.append(chunks)
            total_sents += n_sents
        elif config.window_type == "sent":
            chunks, n_sents = create_chunks_for_instance_sent(doc_inputs, config.kernel_size, config.stride, True)
            batch_chunks.append(chunks)
            total_sents += n_sents
    return batch_chunks, total_sents


def overlap_score(hyp, ref, c=0.8, k=512, flag=True):
    r = c / k
    ref_2gram = get_n_gram(ref, 2)
    total = len(ref_2gram)
    if total == 0:
        return 0.0
    score = 0
    for i, bi_gram in enumerate(zip(hyp, hyp[1:])):
        w = 1 - i * r
        if bi_gram in ref_2gram:
            score += w
            if flag:
                ref_2gram.remove(bi_gram)
    return score / total


def n_gram_overlap_ratio(hyp, ref, n=2):
    hyp = splitter(Tokenizer.decode(hyp).lower())
    ref = splitter(Tokenizer.decode(ref).lower())

    ref_n_gram = set(get_n_gram(ref, n))
    if len(ref_n_gram) == 0:
        return 0
    inter = (ref_n_gram & set(get_n_gram(hyp, n))) - stop_bigrams
    return len(inter) / len(ref_n_gram)


def n_gram_overlap_count(hyp, ref, n=2):
    hyp = splitter(Tokenizer.decode(hyp).lower())
    ref = splitter(Tokenizer.decode(ref).lower())

    return len((set(get_n_gram(ref, n)) & set(get_n_gram(hyp, n))) - stop_bigrams)


def calculate_overlap_score_full(batch_chunks, batch_refs, flag):
    s = []
    for chunks, refs in zip(batch_chunks, batch_refs):
        n = len(chunks)
        score = []
        for i in range(n):
            score.append(overlap_score(chunks[i], plain_text(refs), flag=flag))
        s.append(score)
    return s


def calculate_overlap_score_sent(batch_chunks, batch_refs, flag):
    s = []
    for chunks, refs in zip(batch_chunks, batch_refs):
        n = len(chunks)
        m = len(refs)
        scores = []
        for i in range(m):
            scores_i = []
            for j in range(n):
                scores_i.append(overlap_score(chunks[j], refs[i], flag=flag))
            scores.append(scores_i)
        s.append(scores)
    return s


def calculate_n_gram_overlap_sent(batch_chunks, batch_refs):
    s = []
    for chunks, refs in zip(batch_chunks, batch_refs):
        n = len(chunks)
        m = len(refs)
        scores = []
        for i in range(m):
            scores_i = []
            for j in range(n):
                scores_i.append(n_gram_overlap_count(chunks[j], refs[i]))
            scores.append(cp(scores_i))
        s.append(scores)
    return s


def pick_golden_chunks(batch_chunks, batch_refs, require_batch_qualify=False):
    batch_n_gram_overlap = calculate_n_gram_overlap_sent(batch_chunks, batch_refs)

    ret = []
    if require_batch_qualify:
        ret_qualify = []

    for inst_n_gram_overlap in batch_n_gram_overlap:
        picked_ids_ = list(np.argmax(np.asarray(inst_n_gram_overlap), axis=-1))
        qualify = [(max(it) >= 4) for it in inst_n_gram_overlap]
        cur = 0
        picked_ids = []
        for qual, pick in zip(qualify, picked_ids_):
            if qual:
                cur = pick
            picked_ids.append(cur)
        ret.append(cp(picked_ids))
        if require_batch_qualify:
            ret_qualify.append(cp(qualify))
    if require_batch_qualify:
        return ret, ret_qualify
    return ret


def apply_filter(inputs, outputs, config):
    batch_chunks = create_chunks_for_batch([inputs], config)
    ref_chunk_ids = pick_golden_chunks(batch_chunks, [outputs])

    ref_chunk_ids = ref_chunk_ids[0]

    hyp = []

    for chunk_id in sorted(list(set(ref_chunk_ids))):
        if chunk_id >= 0:
            hyp += batch_chunks[0][chunk_id]

    ref = plain_text(outputs)

    n_overlap = n_gram_overlap_count(hyp, ref)
    return n_overlap < config.n_overlap_constrain


def pick_first_chunk(batch_scores):
    ret = []
    for inst_scores in batch_scores:
        picked_id = int(np.argmax(np.asarray(inst_scores)))
        ret.append(picked_id)
    return ret


def get_batch_chunks_by_ids(batch_chunks, batch_golden_chunk_ids):
    batch_golden_chunks = []
    for chunks, ids in zip(batch_chunks, batch_golden_chunk_ids):
        golden_chunks = []
        if type(ids) == list:
            for id in ids:
                golden_chunks.append(cp(chunks[id]))
        else:
            golden_chunks.append(cp(chunks[ids]))
        batch_golden_chunks.append(golden_chunks)
    return batch_golden_chunks


def prepare_offline_data(batch_data, config):
    batch_docs, batch_refs, batch_indices = batch_data
    batch_chunks = create_chunks_for_batch(batch_docs, config)
    batch_lengths = [len(it) + 2 for it in batch_chunks]
    batch_scores = calculate_overlap_score_full(batch_chunks, batch_refs, True)

    batch_chunks = adding_cls_sep_3d(batch_chunks, config.cls, config.sep)
    padded_batch_chunks = pad_3d_inputs(batch_chunks, config.pad)

    return padded_batch_chunks, batch_indices, batch_lengths, batch_scores


def prepare_ext_data(batch_data, config, h=None):
    """
        doc_inputs: bsz, n, length
        ref_inputs: bsz, m, length
    """
    batch_docs, batch_refs, batch_indices = batch_data
    batch_chunks = create_chunks_for_batch(batch_docs, config)
    batch_scores = calculate_overlap_score_full(batch_chunks, batch_refs, True)

    batch_chunks = adding_cls_sep_3d(batch_chunks, config.cls, config.sep)
    padded_batch_chunks = pad_3d_inputs(batch_chunks, config.pad)

    batch_scores = pad_2d_inputs(batch_scores, -1, t=torch.FloatTensor)
    if h is None:
        return padded_batch_chunks, batch_scores, batch_indices

    batch_chunk_hidden = []
    for id in batch_indices:
        batch_chunk_hidden.append(h[id])
    bsz = len(batch_indices)
    max_l = max([h.shape[0] for h in batch_chunk_hidden])
    padded_batch_chunk_hidden = torch.zeros((bsz, max_l, config.doc_encoder.d_model))
    padded_batch_chunk_attention_mask = torch.zeros((bsz, max_l, max_l))
    for i, chunk_hidden in enumerate(batch_chunk_hidden):
        l = chunk_hidden.shape[0]
        padded_batch_chunk_hidden[i, :l] = torch.from_numpy(chunk_hidden)
        padded_batch_chunk_attention_mask[i, :l, :l] = torch.ones(l, l)
    return padded_batch_chunk_hidden, batch_scores, padded_batch_chunk_attention_mask


def prepare_cross_attention_masks(batch_golden_chunks, batch_refs, config):
    """
        in each instance golden_chunks[0] is the first chunk (C1*)

        # [cls] Sent1 Sent2 [sep]
        #   C1*   C1*    C2    NA
    """
    bsz = len(batch_refs)
    max_num = max([len(golden_chunks) for golden_chunks in batch_golden_chunks])
    max_src_len = max([len(ck) for gck in batch_golden_chunks for ck in gck])
    max_tgt_len = max([sum([len(ref) for ref in refs]) + 1 for refs in batch_refs])
    batch_att_masks = torch.zeros((bsz, max_tgt_len, max_src_len * max_num))
    for i, (golden_chunks, refs) in enumerate(zip(batch_golden_chunks, batch_refs)):
        num = len(golden_chunks)
        tgt_len = sum([len(ref) for ref in refs]) + 1
        src_len = max_src_len * len(golden_chunks)
        att_mask = torch.zeros(tgt_len, src_len)

        p_tgt = 0
        for j, (chunk, ref) in enumerate(zip(golden_chunks, refs)):
            l_src = len(chunk)
            l_tgt = len(ref)
            att_mask[p_tgt:p_tgt + l_tgt, j * max_src_len: j * max_src_len + l_src] = torch.ones(l_tgt, l_src)
            p_tgt += l_tgt

        # last step:
        pt = (num - 1) * max_src_len
        att_mask[-1:, pt: pt + len(golden_chunks[-1])] = torch.ones(1, len(golden_chunks[-1]))

        batch_att_masks[i, :tgt_len, :src_len] = att_mask
    return batch_att_masks


def prepare_cross_attention_masks_test(batch_golden_chunks, batch_refs, config):
    """
        in each instance golden_chunks[0] is the first chunk (C1*)

        # [cls] Sent1 Sent2 [sep]
        #   C1*   C1*    C2    NA
    """
    bsz = len(batch_refs)
    max_num = max([len(golden_chunks) for golden_chunks in batch_golden_chunks])
    max_src_len = max([len(ck) for gck in batch_golden_chunks for ck in gck])
    max_tgt_len = max([sum([len(ref) for ref in refs]) for refs in batch_refs])
    batch_att_masks = torch.zeros((bsz, max_tgt_len, max_src_len * max_num))

    for i, (golden_chunks, refs) in enumerate(zip(batch_golden_chunks, batch_refs)):
        tgt_len = sum([len(ref) for ref in refs])
        src_len = max_src_len * len(golden_chunks)
        att_mask = torch.zeros(tgt_len, src_len)

        p_tgt = 0
        for j, (chunk, ref) in enumerate(zip(golden_chunks, refs)):
            l_src = len(chunk)
            l_tgt = len(ref)
            att_mask[p_tgt:p_tgt + l_tgt, j * max_src_len: j * max_src_len + l_src] = torch.ones(l_tgt, l_src)
            p_tgt += l_tgt

        batch_att_masks[i, :tgt_len, :src_len] = att_mask
    return batch_att_masks


def prepare_switch_label(batch_refs, config):
    bsz = len(batch_refs)
    max_tgt_len = max([sum([len(ref) for ref in refs]) + 1 for refs in batch_refs])

    switch_labels = torch.zeros(bsz, max_tgt_len)
    for i, refs in enumerate(batch_refs):
        labels = []
        for ref in refs:
            labels += [0] * (len(ref) - 1) + [1]
        labels += [0]
        switch_labels[i, : len(labels)] = torch.FloatTensor(labels)
    return switch_labels


def extend1d(inputs):
    return [[it] for it in inputs]


def prepare_abs_data(batch_data, config):
    batch_docs, batch_refs, __ = batch_data
    batch_chunks = create_chunks_for_batch(batch_docs, config)

    batch_golden_chunk_ids = pick_golden_chunks(batch_chunks, batch_refs)

    batch_golden_chunk_ids = [[0] + golden_chunk_ids[1:] for golden_chunk_ids in batch_golden_chunk_ids]

    batch_golden_chunks = get_batch_chunks_by_ids(batch_chunks, batch_golden_chunk_ids)
    batch_golden_chunks = adding_cls_sep_3d(batch_golden_chunks, config.cls, config.sep)

    # batch_cross_attention_mask: bsz, tgt_len, src_len
    batch_cross_attention_mask = prepare_cross_attention_masks(batch_golden_chunks, batch_refs, config)

    batch_plain_refs = [plain_text(refs) for refs in batch_refs]
    batch_plain_refs = adding_cls_sep_2d(batch_plain_refs, config.cls, config.sep)

    batch_golden_chunks = pad_3d_inputs(batch_golden_chunks, config.pad)
    batch_plain_refs = pad_2d_inputs(batch_plain_refs, config.pad)

    batch_decoder_inputs = batch_plain_refs[:, :-1]
    batch_labels = batch_plain_refs[:, 1:]

    return batch_golden_chunks, batch_decoder_inputs, batch_cross_attention_mask.unsqueeze(1), batch_labels


def prepare_ret_labels(batch_golden_chunk_ids, batch_refs, config):
    retrieval_labels = []
    for golden_ids, refs in zip(batch_golden_chunk_ids, batch_refs):
        labels = []
        for id, ref in zip(golden_ids, refs):
            labels += [id] * len(ref)
        labels += [golden_ids[-1]]
        retrieval_labels.append(labels)
    return retrieval_labels


def prepare_regularizer(batch_chunks, batch_refs, config):
    if config.ret_loss_per_token:
        tgt_lens = [sum([len(ref) for ref in refs]) + 1 for refs in batch_refs]
        batch_indices = [list(range(tgt_len)) for tgt_len in tgt_lens]
        batch_indices_mask = [[[1] * (tgt_len - 1)] for tgt_len in tgt_lens]
    else:
        batch_indices = []
        batch_indices_mask = []
        for refs in batch_refs:
            p = 0
            indices = []
            for ref in refs:
                p += len(ref)
                indices.append(p)
            if len(indices) > 0:
                indices = indices[:-1]
            batch_indices.append(indices)
            if len(indices) > 1:
                batch_indices_mask.append([1] * (len(indices) - 1))
            else:
                batch_indices_mask.append([])

    batch_indices = pad_2d_inputs(batch_indices, 0, torch.LongTensor)
    batch_indices_mask = pad_2d_inputs(batch_indices_mask, 0, torch.FloatTensor)
    batch_norm = torch.relu(torch.sum(batch_indices_mask, dim=-1) - 1)
    return batch_indices, batch_indices_mask, batch_norm


def prepare_ret_data(batch_data, config, h=None):
    batch_docs, batch_refs, batch_indices = batch_data
    batch_chunks = create_chunks_for_batch(batch_docs, config)
    batch_src_mask = [[1] * len(chunks) for chunks in batch_chunks]
    batch_src_mask = pad_2d_inputs(batch_src_mask, 0, torch.FloatTensor)

    # need length for the regulizer
    batch_tgt_indices, batch_tgt_indices_mask, batch_regular_norm = \
        prepare_regularizer(batch_chunks, batch_refs, config)

    batch_golden_chunk_ids = pick_golden_chunks(batch_chunks, batch_refs)

    batch_golden_chunk_ids = [[0] + golden_chunk_ids[1:] for golden_chunk_ids in batch_golden_chunk_ids]

    batch_golden_chunks = get_batch_chunks_by_ids(batch_chunks, batch_golden_chunk_ids)
    batch_golden_chunks = adding_cls_sep_3d(batch_golden_chunks, config.cls, config.sep)

    # batch_cross_attention_mask: bsz, tgt_len, src_len
    batch_cross_attention_mask = prepare_cross_attention_masks(batch_golden_chunks, batch_refs, config)

    # batch_switch_labels
    batch_switch_labels = prepare_switch_label(batch_refs, config)
    retrieval_labels = prepare_ret_labels(batch_golden_chunk_ids, batch_refs, config)
    retrieval_labels = pad_2d_inputs(retrieval_labels, config.ignore_index, torch.LongTensor)

    batch_plain_refs = [plain_text(refs) for refs in batch_refs]
    batch_plain_refs = adding_cls_sep_2d(batch_plain_refs, config.cls, config.sep)

    batch_golden_chunks = pad_3d_inputs(batch_golden_chunks, config.pad)
    batch_plain_refs = pad_2d_inputs(batch_plain_refs, config.pad, torch.LongTensor)

    batch_decoder_inputs = batch_plain_refs[:, :-1]
    batch_labels = batch_plain_refs[:, 1:]

    batch_chunk_hidden = []
    for id in batch_indices:
        batch_chunk_hidden.append(h[id])
    bsz = len(batch_indices)
    max_l = max([h.shape[0] for h in batch_chunk_hidden])
    padded_batch_chunk_hidden = torch.zeros((bsz, max_l, config.doc_encoder.d_model))
    padded_batch_chunk_attention_mask = torch.zeros((bsz, max_l, max_l))
    for i, chunk_hidden in enumerate(batch_chunk_hidden):
        l = chunk_hidden.shape[0]
        padded_batch_chunk_hidden[i, :l] = torch.from_numpy(chunk_hidden)
        padded_batch_chunk_attention_mask[i, :l, :l] = torch.ones(l, l)

    return padded_batch_chunk_hidden, padded_batch_chunk_attention_mask, \
           batch_golden_chunks, batch_decoder_inputs, batch_cross_attention_mask.unsqueeze(1), batch_labels, \
           batch_switch_labels, retrieval_labels, \
           batch_tgt_indices, batch_tgt_indices_mask, batch_src_mask, batch_regular_norm, \
           batch_golden_chunk_ids


def prepare_swh_data(batch_data, config):
    batch_docs, batch_refs, __ = batch_data
    batch_chunks = create_chunks_for_batch(batch_docs, config)

    batch_golden_chunk_ids = pick_golden_chunks(batch_chunks, batch_refs)

    batch_golden_chunk_ids = [[0] + golden_chunk_ids[1:] for golden_chunk_ids in batch_golden_chunk_ids]

    batch_golden_chunks = get_batch_chunks_by_ids(batch_chunks, batch_golden_chunk_ids)
    batch_golden_chunks = adding_cls_sep_3d(batch_golden_chunks, config.cls, config.sep)

    # batch_cross_attention_mask: bsz, tgt_len, src_len
    batch_cross_attention_mask = prepare_cross_attention_masks(batch_golden_chunks, batch_refs, config)

    # batch_switch_labels
    batch_switch_labels = prepare_switch_label(batch_refs, config)

    batch_plain_refs = [plain_text(refs) for refs in batch_refs]
    batch_plain_refs = adding_cls_sep_2d(batch_plain_refs, config.cls, config.sep)

    batch_golden_chunks = pad_3d_inputs(batch_golden_chunks, config.pad)
    batch_plain_refs = pad_2d_inputs(batch_plain_refs, config.pad)

    batch_decoder_inputs = batch_plain_refs[:, :-1]
    batch_labels = batch_plain_refs[:, 1:]

    return batch_golden_chunks, batch_decoder_inputs, batch_cross_attention_mask.unsqueeze(1), \
           batch_labels, batch_switch_labels


def prepare_data(batch_data, config, h=None):
    if config.mode == "ext":
        return prepare_ext_data(batch_data, config, h)
    elif config.mode == "abs":
        return prepare_abs_data(batch_data, config)
    elif config.mode == "swh":
        return prepare_swh_data(batch_data, config)
    elif config.mode == "ret":
        return prepare_ret_data(batch_data, config, h)


def prepare_data_cuda(batch_data, config, h=None):
    ret = prepare_data(batch_data, config, h)
    cuda_ret = ()
    for it in list(ret):
        if type(it) != list:
            cuda_ret += (it.cuda(config.device),)
        else:
            cuda_ret += (it,)
    return cuda_ret


def prepare_test(batch_data, config):
    batch_docs, _, batch_indices = batch_data
    batch_chunks = create_chunks_for_batch(batch_docs, config)
    return batch_chunks, batch_indices


def prepare_ext_test(batch_indices, config, h):
    batch_chunk_hidden = []
    for id in batch_indices:
        batch_chunk_hidden.append(h[id])
    bsz = len(batch_indices)
    max_l = max([h.shape[0] for h in batch_chunk_hidden])
    padded_batch_chunk_hidden = torch.zeros((bsz, max_l, config.doc_encoder.d_model))
    padded_batch_chunk_attention_mask = torch.zeros((bsz, max_l, max_l))
    for i, chunk_hidden in enumerate(batch_chunk_hidden):
        l = chunk_hidden.shape[0]
        padded_batch_chunk_hidden[i, :l] = torch.from_numpy(chunk_hidden)
        padded_batch_chunk_attention_mask[i, :l, :l] = torch.ones(l, l)
    return padded_batch_chunk_hidden, padded_batch_chunk_attention_mask


def prepare_abs_test(batch_chunks, golden_chunk_ids, summary_sents, config):
    batch_golden_chunks = get_batch_chunks_by_ids(batch_chunks, [golden_chunk_ids])
    batch_golden_chunks = adding_cls_sep_3d(batch_golden_chunks, config.cls, config.sep)

    batch_cross_attention_mask = prepare_cross_attention_masks_test(batch_golden_chunks, [summary_sents], config)
    batch_golden_chunks = pad_3d_inputs(batch_golden_chunks, config.pad)

    batch_plain_sum = [plain_text(summary_sents)]

    batch_decoder_inputs = pad_2d_inputs(batch_plain_sum, config.pad)

    return batch_golden_chunks, batch_decoder_inputs, batch_cross_attention_mask.unsqueeze(1)


def prepare_abs_emb_test(summary_sents, next_token, config):
    shift_token = [plain_text(summary_sents) + [next_token]]
    shift_token = pad_2d_inputs(shift_token, config.pad)
    return shift_token[:, 1:]


def convert_cuda(ret, config):
    if type(ret) == tuple:
        cuda_ret = ()
        for it in list(ret):
            if type(it) != list:
                cuda_ret += (it.cuda(config.device),)
            else:
                cuda_ret += (it,)
        return cuda_ret
    elif type(ret) == list:
        return ret
    else:
        return ret.cuda(config.device)
