import torch

from data_process import MyTokenizer
from data_process import Podcasts as Dataset

from layers import Retrieval
from search.beam import beam_search

from utility import load_from_pkl


def get_data(config, log, mode='test'):
    print("get_data test")
    tokenizer = MyTokenizer(config)
    valid_set = Dataset(
        name=config.test,
        len_func=lambda x: sum(len(it) for it in x[0]),
        config=config,
        tokenizer=tokenizer,
        log=log,
        mode=mode
    )
    n_valid = len(valid_set)
    log.log("There are %d batches in valid data" % n_valid)
    return n_valid, valid_set


def get_network(config, log):
    # Model Setup
    log.log("Building Model")
    net = Retrieval(config)

    # Loading Parameter
    log.log("Loading Parameters")
    best_model = torch.load(config.save_path + "/" + config.model)
    new_stat_dict = {}
    for key, value in best_model["state_dict"].items():
        if key.startswith("module."):
            new_key = key[7:]
        else:
            new_key = key
        new_stat_dict[new_key] = value
    net.load_state_dict(new_stat_dict)
    log.log("Parameters Loaded")

    net = net.cuda(config.device)

    net.eval()
    log.log("Finished Build Model")
    return net


def test(config, log):
    tokenizer = MyTokenizer(config)
    config.batch_size = 1
    config.mode = "ret"
    net = get_network(config, log)
    _, valid_set = get_data(config, log)

    name = "hidden_v_" + str(config.kernel_size) + "_" + str(config.stride) + "_" + config.window_type + ".pkl"
    h_v = load_from_pkl(name)

    suffix = "_" + str(config.stride) + "_" + str(config.beam_size) + "_" + str(config.length_penalty)
    f = open("summary" + suffix + ".txt", "w")

    torch.cuda.empty_cache()

    for batch_idx, batch_data in enumerate(valid_set):
        print(batch_idx)
        answer, _ = beam_search(net, batch_data, config, h=h_v, output_others=True)

        ans = []
        first = True
        for sent in answer[2]:
            if first:
                ans.append(sent)
                first = False
            else:
                ans[-1].append(sent[0])
                ans.append(sent[1:])
        summary_text = " [SSPLIT] ".join(tokenizer.decode(sent) for sent in ans)
        if summary_text.endswith(" [SSPLIT] "):
            summary_text = summary_text[:-10]
        print(summary_text)
        print(summary_text, file=f)

    f.close()
