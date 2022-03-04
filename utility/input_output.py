import json
import pickle
import shutil

import torch


# IO
def load_from_json(filename):
    f = open(filename, 'r', encoding='utf-8')
    data = json.load(f, strict=False)
    f.close()
    return data


def save_to_json(filename, data):
    f = open(filename, 'w', encoding='utf-8')
    json.dump(data, f, indent=4)
    f.close()
    return True


def save_to_pkl(filename, data):
    with open(filename, 'wb')as f:
        pickle.dump(data, f)
    return


def load_from_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def write_file(filename, massage):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(massage)
    return True


def save_check_point(state, is_best, path='.model', file_name='latest.pth.tar', mode=None):
    if mode is None:
        name = path + '/' + file_name
        torch.save(state, name)
        if is_best:
            shutil.copyfile(name, path + '/model_best.pth.tar')
            shutil.copyfile(name, path + '/model_best_epoch_' + str(state['epoch']) + '.pth.tar')
    else:
        name = path + '/' + mode + '_' + file_name
        torch.save(state, name)
        if is_best:
            shutil.copyfile(name, path + '/' + mode + '_model_best.pth.tar')
            shutil.copyfile(name, path + '/' + mode + '_model_best_epoch_' + str(state['epoch']) + '.pth.tar')
