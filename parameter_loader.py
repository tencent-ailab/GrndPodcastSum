import argparse
import os
from argparse import Namespace

import torch

from utility import load_from_json


def arg_loader():
    parser = argparse.ArgumentParser()

    """
        Actions: train/test/debug/offline
        Mode: ext/abs/swh/ret
        Mini: for debugging load less data
    """
    parser.add_argument('--do_train', action='store_true', help="Whether to run training")
    parser.add_argument('--do_test', action='store_true', help="Whether to run test")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--example', action='store_true')
    parser.add_argument('--statistic', action='store_true')
    parser.add_argument('--create_ref', action='store_true')

    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--offline_k', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--ret_loss_per_token', action="store_true")
    parser.add_argument('--ignore_index', type=int, default=-1)
    parser.add_argument('--regularizer', type=int, default=1)
    parser.add_argument('--coef', type=float, default=0.1)

    parser.add_argument('--mini', action="store_true")
    parser.add_argument('--T_p', type=int, default=25)

    parser.add_argument('--Th', type=float, default=0.5)
    parser.add_argument('--cls_dim', type=int, default=256)

    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--overlap', action="store_true")

    """
        Training Tricks:
            amp: Automatic Mixed Precision
            parallel: data-parallel
            device: default device
            n_workers: how many thread for data loading (current unavailable)
    """

    parser.add_argument('--amp', action='store_true', help="Whether or not to use amp during trianing")
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=2)

    """
        Data Source: dataset/raw
            load: whether build from files or load from cache
            train/valid/test: the part of the dataset
            raw_train/valid/test: the path to raw file
            length_limit_input/output: maximum length
    """
    parser.add_argument('--load', action='store_true', help="Whether or not build from raw")

    # From dataset
    parser.add_argument('--from_dataset', action='store_true', help="From dataset")
    parser.add_argument('--dataset', type=str, default='podcasts')
    parser.add_argument('--train', type=str, default='train')
    parser.add_argument('--valid', type=str, default='valid')
    parser.add_argument('--test', type=str, default='test')

    # From Raw Files
    parser.add_argument('--from_raw', action='store_true', help="From raw text")
    parser.add_argument('--raw_train', type=str, default="raw_train.txt")
    parser.add_argument('--raw_valid', type=str, default="raw_valid.txt")
    parser.add_argument('--raw_test', type=str, default="raw_test.txt")

    # Truncated
    parser.add_argument('--input_limit', type=int, default=65536)
    parser.add_argument('--output_limit', type=int, default=128)
    parser.add_argument('--output_sent_limit', type=int, default=8)

    # Constrain
    parser.add_argument('--n_overlap_constrain', type=int, default=3)

    # Pre-ext parameters
    parser.add_argument('--threshold', type=float, default=0.2)

    """
        Pretrained Model Selection:
            --model_ext: roberta-base roberta-large
            --model_abs: facebook/bart-base facebook/bart-large facebook/bart-large-cnn
            --local: whether or not behind a firewall
        Current Version ext_model will takes bart encoder as its sentence encoder
    """
    parser.add_argument('--model_ext', type=str, default='roberta-large')
    parser.add_argument('--model_abs', type=str, default='facebook/bart-large')
    parser.add_argument('--local', action='store_true', help="Whether or not using local models")

    """
        Model Settings:
            Save Path
            Sliding Window
            Sampling (between ext part and abs part when training in mode "ext-abs")
            Extractor
    """
    # Save Path
    parser.add_argument('--main_path', type=str, default='./model')
    parser.add_argument('--abs_path', type=str, default='/abs')
    parser.add_argument('--ext_path', type=str, default='/ext')
    parser.add_argument('--swh_path', type=str, default='/swh')
    parser.add_argument('--ret_path', type=str, default='/ret')

    # Sliding Window
    parser.add_argument('--window_type', type=str, default="sent")
    parser.add_argument('--kernel_size', type=int, default=20)
    parser.add_argument('--stride', type=int, default=10)

    # Model Parameters
    parser.add_argument('--ext_type', type=str, default="TokenLevelEncoder")
    parser.add_argument('--d_query', type=int, default=1024)
    parser.add_argument('--d_key', type=int, default=1024)
    parser.add_argument('--d_att', type=int, default=1024)
    parser.add_argument('--d_inner', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--type_att', type=int, default=3)

    # Transformer Encoder
    parser.add_argument('--doc_n_layers', type=int, default=2)
    parser.add_argument('--doc_d_model', type=int, default=1024)
    parser.add_argument('--doc_n_head', type=int, default=16)
    parser.add_argument('--doc_d_ff', type=int, default=3072)
    parser.add_argument('--doc_max_len', type=int, default=1024)
    parser.add_argument('--doc_dropout', type=float, default=0.1)

    # Loss Parameters
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    """
        Optimization:
            learning rate: 1e-5 (32), 1.224e-5(48), 3.4641e-5(384)
            weight decay
            adam_epsilon
            warmup steps: 5% to 10% of first epoch
    """

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-7)
    parser.add_argument('--warmup_steps', type=int, default=100)

    """
        Training:
            max epoch: 5, 10, 20
            batch_size: 1, 8, 16, 32
            checkPoint_Min/Freq: CheckPoint parameters
            save each epoch: for restore training (usually no need)
    """
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--checkPoint_Min', type=int, default=0)
    parser.add_argument('--checkPoint_Freq', type=int, default=100)
    parser.add_argument('--save_each_epoch', action="store_true")

    # Testting Parameters
    parser.add_argument('--model', type=str, default='model_best.pth.tar')

    # Sentence Decoding Parameters
    parser.add_argument('--gen_max_len', type=int, default=128)
    parser.add_argument('--gen_min_len', type=int, default=20)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--chunk_beam_size', type=int, default=4)
    parser.add_argument('--answer_size', type=int, default=4)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)

    args = parser.parse_args()
    # Build or Load
    args.build = not args.load

    # Special Tokens
    args.pad = 1
    args.UNK = 3
    args.cls = 0
    args.sep = 2
    args.BOS = 50261
    args.EOS = 50260
    args.MASK = 50264
    args.n_vocab = 50265

    # Doc Encoder
    args.doc_encoder = Namespace()
    args.doc_encoder.pre_training_model = args.model_ext
    args.doc_encoder.n_layers = args.doc_n_layers
    args.doc_encoder.d_model = args.doc_d_model
    args.doc_encoder.n_head = args.doc_n_head
    args.doc_encoder.d_ff = args.doc_d_ff
    args.doc_encoder.max_len = args.doc_max_len
    args.doc_encoder.dropout = args.doc_dropout
    args.doc_encoder.pad = args.pad
    args.doc_encoder.autoregressive = False

    # Local Pre-trained Models
    if args.local:
        args.model_ext = "../../pretrained_models/" + args.model_ext
        args.model_abs = "../../pretrained_models/" + args.model_abs

    # Data Loading options
    args.dataOptions = load_from_json("settings/dataset/" + str(args.dataset) + ".json")
    args.strategy = None

    # Model Save Path
    if args.mode == "ext":
        args.save_path = args.main_path + args.ext_path
    elif args.mode == "abs":
        args.save_path = args.main_path + args.abs_path
    elif args.mode == "swh":
        args.save_path = args.main_path + args.swh_path
    elif args.mode == "ret":
        args.save_path = args.main_path + args.ret_path
    else:
        args.save_path = args.main_path + "/ret-" + str(args.stride) + "-01"
    # Make Dirs
    if not os.path.exists(args.main_path):
        os.makedirs(args.main_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # tricks
    # args.triGramTrick = not args.no_triGramTrick

    # Threshold
    if args.mode == "ext" and (args.do_train or args.do_test):
        name = "parts_" + str(args.kernel_size) + "_" + str(args.stride) + "_" + args.window_type + ".pt"
        parts = torch.load(name)
        args.T = list(parts.values())[args.T_p // 5 - 1]
        print(args.T)

    # print(args)
    return args
