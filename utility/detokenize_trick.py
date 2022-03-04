from copy import deepcopy as cp


def mapping_tokenize(s, t):
    st = 0
    ed = 0
    mapping = []
    mapping_idx = []
    for idx, token in enumerate(s):
        token_ = token.lower()
        prefix = "".join([piece.replace('##', '') for piece in t[st:ed + 1]])
        while token_.startswith(prefix):
            ed += 1
            if ed >= len(t):
                break
            prefix = "".join([piece.replace('##', '') for piece in t[st:ed + 1]])
        if (ed - st > 1) or (sum(1 for c in token if c.isupper()) > 1) or (idx > 0):
            mapping_idx.append([(st, ed), idx])
            mapping.append([cp(t[st:ed]), token])
        st = ed
    return mapping


def detokenize(text, mapping):
    if mapping is None:
        return text
    text = " " + text
    for one_mapping in mapping:
        keys = "".join([key.replace('##', '') if key.startswith('##') else ' ' + key for key in one_mapping[0]])
        value = ' ' + one_mapping[1]
        text = text.replace(keys, value)
    text = list(text[1:])
    if len(text) > 0:
        text[0] = text[0].upper()
        text = "".join(text)
    return text
