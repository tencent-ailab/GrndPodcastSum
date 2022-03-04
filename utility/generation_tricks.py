def get_n_gram(seq, n):
    return list(zip(*[seq[i:] for i in range(n)]))


def do_tricks(preds, source, target, config):
    ban_ids = []

    # n_gram_blocking
    if config.no_repeat_ngram_size == 1:
        ban_ids = list(set(target))
    if (config.no_repeat_ngram_size > 1) and (len(target) >= config.no_repeat_ngram_size):
        current_n_grams = get_n_gram(target, config.no_repeat_ngram_size)
        for n_gram in current_n_grams:
            if all(t_token == ng_token for t_token, ng_token in zip(target[1 - config.no_repeat_ngram_size:], n_gram)):
                ban_ids.append(n_gram[-1])

    # min_length
    if len(target) < config.gen_min_len:
        ban_ids.append(config.sep)

    for idx in ban_ids:
        preds[idx] = -float("inf")

    # blocking NAN and INF
    preds[preds != preds] = -1e9
    preds[preds == float("inf")] = -1e9

    return preds


def trigram_blocking(preds, target, config):
    ban_ids = []
    if config.triGramTrick and len(target) > 2:
        current_tri_grams = get_n_gram(target, 3)
        for tri_gram in current_tri_grams:
            if (target[-2] == tri_gram[0]) and (target[-1] == tri_gram[1]):
                ban_ids.append(tri_gram[2])

    for idx in ban_ids:
        preds[idx] = -1e9

    return preds
