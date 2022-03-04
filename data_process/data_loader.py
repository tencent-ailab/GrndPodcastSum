import json
import random

import numpy as np
import torch
import torch.utils.data as Data

from utility import load_from_pkl, save_to_pkl, apply_filter


class MyDataset(Data.Dataset):
    def __init__(self, name, len_func, config, log, mode='train', prepare_func=None):
        self.name = name
        self.log = log
        self.len_func = len_func
        self.mode = mode
        self.prepare_func = prepare_func
        self.config = config

        self.build = config.build
        self.batch_size = config.batch_size

        self.path = config.dataOptions['Parts'][name]['path']
        self.sorted = config.dataOptions['Parts'][name]['sorted']
        self.shuffled = config.dataOptions['Parts'][name]['shuffled']

        self.n_data = 0
        self.data = []
        self.n_batch = 0
        self.batch = []
        self.batch_idx = []

    def sort_by_length(self):
        self.log.log('Start sorting by length')
        data = self.data
        number = self.n_data

        lengths = [(self.len_func(data[Index]), Index) for Index in range(number)]
        sorted_lengths = sorted(lengths)
        sorted_index = [d[1] for d in sorted_lengths]

        data_new = [data[sorted_index[Index]] for Index in range(number)]

        self.data = data_new
        self.log.log('Finish sorting by length')

    def shuffle(self):
        self.log.log('Start Shuffling')

        data = self.data
        number = self.n_data

        shuffle_index = list(range(number))
        random.shuffle(shuffle_index)

        data_new = [data[shuffle_index[Index]] for Index in range(number)]

        self.data = data_new
        self.log.log('Finish Shuffling')

    def gen_batches(self):
        batch_size = self.batch_size
        data = self.data
        number = self.n_data
        n_dim = len(data[0])

        number_batch = number // batch_size
        batches = []

        for bid in range(number_batch):
            batch_i = []
            for j in range(n_dim):
                data_j = [item[j] for item in data[bid * batch_size: (bid + 1) * batch_size]]
                batch_i.append(data_j)
            batches.append(batch_i)

        if number_batch * batch_size < number:
            if number - number_batch * batch_size >= torch.cuda.device_count():
                batch_i = []
                for j in range(n_dim):
                    data_j = [item[j] for item in data[number_batch * batch_size:]]
                    batch_i.append(data_j)
                batches.append(batch_i)
                number_batch += 1

        self.n_batch = number_batch
        self.batch = batches
        self.batch_idx = list(range(self.n_batch))

    def load(self):
        pass

    def after_load(self):
        if (self.mode != "test") and self.sorted:
            self.sort_by_length()
        if (self.mode != "test") and self.shuffled:
            self.shuffle()

        # Generate Batches
        self.log.log('Generating Batches')
        self.gen_batches()

        for_save = {
            "n_data": self.n_data,
            "Data": self.data,
            "n_batch": self.n_batch,
            "Batch": self.batch,
            "Batch_idx": self.batch_idx
        }

        save_to_pkl(self.name + ".cache", for_save)

    def batch_shuffle(self):
        random.shuffle(self.batch_idx)

    def __len__(self):
        if self.mode == 'train' or self.mode == 'valid':
            return self.n_batch
        return self.n_data

    def __getitem__(self, index):
        if self.prepare_func is None:
            return self.batch[self.batch_idx[index]]
        return self.prepare_func(self.batch[self.batch_idx[index]])


class Podcasts(MyDataset):
    def __init__(self, name, len_func, tokenizer, config, log, mode='train', prepare_func=None):
        super(Podcasts, self).__init__(name, len_func, config, log, mode, prepare_func)
        self.mini = config.mini
        self.input_limit = config.input_limit
        self.output_limit = config.output_limit
        self.kernel_size = config.kernel_size
        self.stride = config.stride

        self.tokenizer = tokenizer
        self.train_mode = config.mode
        self.threshold = config.threshold
        self.mode = mode

        # Loading Dataset
        if self.build:
            self.log.log('Building dataset %s from orignial text documents' % self.name)
            self.n_data, self.data = self.load()
            self.after_load()
            self.log.log('Finish Loading dataset %s' % self.name)
        else:
            self.log.log("Loading dataset %s from cached files" % self.name)
            for_load = load_from_pkl(self.name + ".cache")
            self.n_data = for_load["n_data"]
            self.data = for_load["Data"]
            self.n_batch = for_load["n_batch"]
            self.batch = for_load["Batch"]
            self.batch_idx = for_load["Batch_idx"]
            self.log.log('Finish Loading dataset %s' % self.name)

    @staticmethod
    def truncate(inputs, front, rear):
        if len(inputs) > front + rear:
            new_inputs = inputs[:front]
            if rear > 0:
                new_inputs += inputs[-rear:]
        else:
            new_inputs = inputs
        return new_inputs

    def load(self):
        input_file = open(self.path + ".json", "r", encoding='utf-8')
        data = []

        f = open(self.name + "_example.txt", "w")

        for index, line in enumerate(input_file):
            if self.mini and (index >= 5000):
                break
            data_i = json.loads(line)
            idx = data_i["episode_uri"]
            inputs_ = data_i["input"]
            outputs_ = data_i["output"]

            inputs = [self.tokenizer.encode(seg) for seg in inputs_]
            outputs = [self.tokenizer.encode(seq) for seq in outputs_]

            if len(inputs) == 0:
                print("Error: Input Empty at", index)
                continue

            if len(outputs) == 0 and self.mode != "test":
                print("Error: Output Empty at", index)
                continue

            lo = sum([len(it) for it in outputs])
            if lo < 10 and self.mode != "test":
                print("Error: Output is too short (%d < 10)." % lo, index)
                continue

            output_lengths = np.asarray([len(seq) for seq in outputs]).cumsum()
            t = 0
            while t + 1 < len(outputs) and output_lengths[t + 1] <= self.output_limit:
                t += 1

            outputs = outputs[:t + 1]

            if self.mode != "test" and apply_filter(inputs, outputs, self.config):
                print("Filter: Instance has too little overlap.", index)
                continue

            data.append([inputs, outputs, idx])

        f.close()
        return len(data), data
