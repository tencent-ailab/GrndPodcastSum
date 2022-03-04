import torch

from mylog import MyLog
from offline_encoding import offline
from parameter_loader import arg_loader
from test import test
from train import train_ext, train_abs, train_swh, train_ret

LOG = MyLog(reset=False)
torch.cuda.set_device(1)


def main():
    config = arg_loader()
    if config.do_train:
        if config.mode == "ext":
            train_ext(config, LOG)
        elif config.mode == "abs":
            train_abs(config, LOG)
        elif config.mode == "swh":
            train_swh(config, LOG)
        elif config.mode == "ret":
            train_ret(config, LOG)
    elif config.do_test:
        test(config, LOG)
    elif config.offline:
        offline(config, LOG)

if __name__ == "__main__":
    main()
