import gc
import time

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from data_process import MyTokenizer
from data_process import Podcasts as Dataset
from layers import KLDivLoss, BCEWithLogitsLoss, CrossEntropyWithRegularizer
from layers import Retrieval
from utility import DataParallelModel, DataParallelCriterion, save_check_point
from utility import prepare_data_cuda, load_from_pkl


class LossWatcher(object):
    def __init__(self, log, max_len=50):
        self.log = log
        self.q = []
        self.max_len = max_len

    def append(self, value):
        self.q.append(value)
        if len(self.q) > self.max_len:
            self.q.pop(0)

    def write_log(self, eid, bid, duration, mode=None):
        if mode is not None:
            self.log.log('Mode %s, Epoch %2d, Batch %6d, Loss %9.6f, Average Loss %9.6f, Time %9.6f' %
                         (mode, eid + 1, bid + 1, self.q[-1], sum(self.q) / len(self.q), duration))
        else:
            self.log.log('Epoch %2d, Batch %6d, Loss %9.6f, Average Loss %9.6f, Time %9.6f' %
                         (eid + 1, bid + 1, self.q[-1], sum(self.q) / len(self.q), duration))


class CheckPoint(object):
    def __init__(self, config, log):
        self.log = log
        self.tick = 0
        self.check_min = config.checkPoint_Min
        self.check_freq = config.checkPoint_Freq
        self.is_checkpoint = False
        self.best_loss = 1e9

    def ticktock(self):
        self.tick += 1
        self.is_checkpoint = (self.tick >= self.check_min) and (self.tick % self.check_freq == 0)
        if self.is_checkpoint:
            gc.collect()
        return self.is_checkpoint

    def update(self, loss, net, eid, bid, config, mode=None):
        self.log.log('CheckPoint: Validation Loss %11.8f, Best Loss %11.8f' % (loss, self.best_loss))
        is_best = loss < self.best_loss
        if is_best:
            self.log.log("Model Update")
            self.best_loss = loss

            save_check_point({
                'epoch': eid,
                'batch': bid,
                'config': config,
                'state_dict': net.state_dict(),
                'best_vloss': self.best_loss},
                is_best,
                path=config.save_path,
                file_name='latest.pth.tar',
                mode=mode
            )


def get_data(config, log):
    tokenizer = MyTokenizer(config)
    train_set = Dataset(
        name=config.train,
        len_func=lambda x: sum(len(it) for it in x[0]),
        config=config,
        tokenizer=tokenizer,
        log=log,
        mode='train'
    )

    valid_set = Dataset(
        name=config.valid,
        len_func=lambda x: sum(len(it) for it in x[0]),
        config=config,
        tokenizer=tokenizer,
        log=log,
        mode='valid'
    )
    n_train = len(train_set)
    n_valid = len(valid_set)
    log.log("There are %d batches in train data" % n_train)
    log.log("There are %d batches in valid data" % n_valid)
    return n_train, train_set, n_valid, valid_set


def get_optimizer(net, config, n_batches, norm=1.0):
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate * norm, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=n_batches * config.max_epoch
    )
    optimizer.zero_grad()

    return optimizer, scheduler


def backward(loss, optimizer, scheduler, scaler=None, log=None):
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
        scheduler.step()

    optimizer.zero_grad()


def get_network(config, log):
    # Build Model
    log.log("Building Model")
    net = Retrieval(config)
    if config.mode == "ext":
        loss_func = BCEWithLogitsLoss()
    elif config.mode == "abs":
        loss_func = KLDivLoss(config)
    elif config.mode == "swh":
        loss_func = BCEWithLogitsLoss()
    elif config.mode == "ret":
        loss_func = CrossEntropyWithRegularizer(config)

    # Parameter Loading
    if config.mode in ["swh", "ret"]:
        log.log("Loading Parameters")
        # Extractor Parameters
        best_ext_model = torch.load(config.main_path + config.ext_path + "/model_best.pth.tar")
        ext_state_dict = {}
        for key, value in best_ext_model["state_dict"].items():
            if key.startswith("module.extractor."):
                ext_state_dict[key[7:]] = value
            elif key.startswith("extractor."):
                ext_state_dict[key] = value
        __, __ = net.load_state_dict(ext_state_dict, strict=False)
        log.log("Extractor Parameters Loaded")

        best_abs_model = torch.load(config.main_path + config.abs_path + "/model_best.pth.tar")
        abs_state_dict = {}
        for key, value in best_abs_model["state_dict"].items():
            if key.startswith("module.abstracter."):
                abs_state_dict[key[7:]] = value
            elif key.startswith("abstracter."):
                abs_state_dict[key] = value
        __, __ = net.load_state_dict(abs_state_dict, strict=False)
        log.log("Abstracter Parameters Loaded")
        if config.mode == "ret":
            # Switch Parameters
            best_swh_model = torch.load(config.main_path + config.swh_path + "/model_best.pth.tar")
            swh_state_dict = {}
            for key, value in best_swh_model["state_dict"].items():
                if key.startswith("module.switch."):
                    swh_state_dict[key[7:]] = value
                elif key.startswith("switch."):
                    swh_state_dict[key] = value
            __, __ = net.load_state_dict(swh_state_dict, strict=False)
            log.log("Switch Parameters Loaded")

    # Move Network and Loss Functions to CUDA
    if torch.cuda.is_available():
        log.log("Using GPU")
        log.log("Totally %d GPUs are available" % torch.cuda.device_count())
        log.log("Moving Network and Loss Function to GPU")
        net = net.cuda(config.device)
        loss_func = loss_func.cuda(config.device)

        if config.parallel:
            log.log("Using data parallel")
            for device in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(device)
                memory = (torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(
                    device)) // 1024 // 1024 / 1024
                log.log("Using # %d GPU, named %s, has %.4f GB Memory available" % (device, name, memory))
            net = DataParallelModel(net)
            loss_func = DataParallelCriterion(loss_func)

        else:
            log.log("Using Single GPU")
            torch.cuda.set_device(config.device)
            name = torch.cuda.get_device_name(config.device)
            memory = (torch.cuda.get_device_properties(config.device).total_memory - torch.cuda.memory_allocated(
                config.device)) // 1024 // 1024 / 1024
            log.log("Using # %d GPU, named %s, has %.4f GB Memory available" % (config.device, name, memory))

    # Setup Training Stages
    if config.parallel:
        net.module.set_mode(config.mode)
    else:
        net.set_mode(config.mode)

    # Setup Training
    net.train()
    log.log("Finished Build Model")
    return net, loss_func


# Step 1 Pre-train Extractor
def train_ext(config, log):
    # Automatic Mixed Precision
    scaler = torch.cuda.amp.GradScaler()
    net, loss_func = get_network(config, log)
    n_train, train_set, __, valid_set = get_data(config, log)
    name = str(config.kernel_size) + "_" + str(config.stride) + "_" + config.window_type + "_align.pkl"
    h = load_from_pkl("./hidden_" + name)
    h_v = load_from_pkl("./hidden_v_" + name)
    optimizer, scheduler = get_optimizer(net, config, n_train)
    watcher = LossWatcher(log)
    checker = CheckPoint(config, log)

    for epoch_idx in range(config.max_epoch):
        train_set.batch_shuffle()
        log.log("Batch Shuffled")
        for batch_idx, batch_data in enumerate(train_set):
            start_time = time.time()
            padded_batch_chunk_hidden, batch_scores, padded_batch_chunk_attention_mask = \
                prepare_data_cuda(batch_data, config, h=h)

            labels = (batch_scores >= config.T).float()
            mask = (batch_scores >= 0).float()

            # Forward & Backward
            predicts = net(
                chunk_hidden=padded_batch_chunk_hidden,
                chunk_attention_mask=padded_batch_chunk_attention_mask
            )

            loss = loss_func(predicts, labels, mask).sum() / mask.sum()

            backward(loss=loss, optimizer=optimizer, scheduler=scheduler, scaler=scaler if config.amp else None)

            # Loss Watcher
            watcher.append(float(loss))
            watcher.write_log(epoch_idx, batch_idx, time.time() - start_time)

            if checker.ticktock():
                net.eval()
                loss = 0
                total = 0
                with torch.no_grad():
                    for batch_data in valid_set:
                        padded_batch_chunk_hidden, batch_scores, padded_batch_chunk_attention_mask = \
                            prepare_data_cuda(batch_data, config, h=h_v)
                        labels = (batch_scores >= config.T).float()
                        mask = (batch_scores >= 0).float()

                        predicts = net(
                            chunk_input_ids=None,
                            chunk_hidden=padded_batch_chunk_hidden,
                            chunk_attention_mask=padded_batch_chunk_attention_mask
                        )
                        loss += loss_func(predicts, labels, mask).sum()
                        total += float(mask.sum())

                if total < 1e-6:
                    loss = 0
                else:
                    loss /= total
                checker.update(loss, net, epoch_idx, batch_idx, config)

        if config.save_each_epoch:
            log.log('Saving Model after %d-th Epoch.' % (epoch_idx + 1))
            save_check_point({
                'epoch': epoch_idx,
                'batch': batch_idx,
                'config': config,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_vloss': 1e99},
                False,
                path=config.save_path,
                file_name='checkpoint_epoch_' + str(epoch_idx) + '.pth.tar'
            )

        log.log('Epoch Finished.')
        gc.collect()


# Step 2 Pre-train Abstracter
def train_abs(config, log):
    # Automatic Mixed Precision
    scaler = torch.cuda.amp.GradScaler()
    net, loss_func = get_network(config, log)
    n_train, train_set, __, valid_set = get_data(config, log)
    optimizer, scheduler = get_optimizer(net, config, n_train)
    watcher = LossWatcher(log)
    checker = CheckPoint(config, log)

    for epoch_idx in range(config.max_epoch):
        train_set.batch_shuffle()
        log.log("Batch Shuffled")
        for batch_idx, batch_data in enumerate(train_set):
            start_time = time.time()
            encoder_input_ids, decoder_input_ids, cross_attention_mask, labels = \
                prepare_data_cuda(batch_data, config)

            # Forward & Backward
            predicts = net(
                input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_attention_mask=cross_attention_mask
            )

            n_token = float((labels.data != config.pad).sum())
            loss = loss_func(predicts, labels, n_token).sum()

            backward(loss=loss, optimizer=optimizer, scheduler=scheduler, scaler=scaler if config.amp else None)

            # Loss Watcher
            watcher.append(float(loss))
            watcher.write_log(epoch_idx, batch_idx, time.time() - start_time)

            if checker.ticktock():
                net.eval()
                loss = 0
                total = 0
                with torch.no_grad():
                    for batch_data in valid_set:
                        encoder_input_ids, decoder_input_ids, cross_attention_mask, labels = \
                            prepare_data_cuda(batch_data, config)

                        predicts = net(
                            input_ids=encoder_input_ids,
                            decoder_input_ids=decoder_input_ids,
                            encoder_attention_mask=cross_attention_mask
                        )

                        n_token = float((labels.data != config.pad).sum())
                        loss += float(loss_func(predicts, labels).sum())
                        total += n_token

                if total < 1e-6:
                    loss = 0
                else:
                    loss /= total
                checker.update(loss, net, epoch_idx, batch_idx, config)

        if config.save_each_epoch:
            log.log('Saving Model after %d-th Epoch.' % (epoch_idx + 1))
            save_check_point({
                'epoch': epoch_idx,
                'batch': batch_idx,
                'config': config,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_vloss': 1e99},
                False,
                path=config.save_path,
                file_name='checkpoint_epoch_' + str(epoch_idx) + '.pth.tar'
            )

        log.log('Epoch Finished.')
        gc.collect()


# Step 3 train switch model
def train_swh(config, log):
    # Automatic Mixed Precision
    scaler = torch.cuda.amp.GradScaler()
    net, loss_func = get_network(config, log)
    n_train, train_set, __, valid_set = get_data(config, log)
    optimizer, scheduler = get_optimizer(net, config, n_train)
    watcher = LossWatcher(log)
    checker = CheckPoint(config, log)

    for epoch_idx in range(config.max_epoch):
        train_set.batch_shuffle()
        log.log("Batch Shuffled")
        for batch_idx, batch_data in enumerate(train_set):
            start_time = time.time()
            golden_chunks, decoder_inputs, cross_attention, abs_labels, switch_labels = \
                prepare_data_cuda(batch_data, config)

            # Forward & Backward
            predict = net(
                input_ids=golden_chunks,
                decoder_input_ids=decoder_inputs,
                encoder_attention_mask=cross_attention,
                decoder_labels=abs_labels,
            )
            mask = (abs_labels.data != config.pad).float()
            n_token = float(mask.sum())
            loss = loss_func(predict, switch_labels, mask, n_token).sum()

            backward(loss=loss, optimizer=optimizer, scheduler=scheduler, scaler=scaler if config.amp else None)

            # Loss Watcher
            watcher.append(float(loss))
            watcher.write_log(epoch_idx, batch_idx, time.time() - start_time)

            if checker.ticktock():
                net.eval()
                loss = 0
                total = 0
                with torch.no_grad():
                    for batch_data in valid_set:
                        golden_chunks, decoder_inputs, cross_attention, abs_labels, switch_labels = \
                            prepare_data_cuda(batch_data, config)

                        # Forward & Backward
                        predict = net(
                            input_ids=golden_chunks,
                            decoder_input_ids=decoder_inputs,
                            encoder_attention_mask=cross_attention,
                            decoder_labels=abs_labels,
                        )
                        mask = (abs_labels.data != config.pad).float()
                        total += float(mask.sum())
                        loss += float(loss_func(predict, switch_labels, mask).sum())
                if total < 1e-6:
                    loss = 0
                else:
                    loss /= total
                checker.update(loss, net, epoch_idx, batch_idx, config)

        if config.save_each_epoch:
            log.log('Saving Model after %d-th Epoch.' % (epoch_idx + 1))
            save_check_point({
                'epoch': epoch_idx,
                'batch': batch_idx,
                'config': config,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_vloss': 1e99},
                False,
                path=config.save_path,
                file_name='checkpoint_epoch_' + str(epoch_idx) + '.pth.tar'
            )

        log.log('Epoch Finished.')
        gc.collect()


# Step 4 train retrieval model
def train_ret(config, log):
    # Automatic Mixed Precision
    scaler = torch.cuda.amp.GradScaler()
    net, loss_func = get_network(config, log)
    n_train, train_set, __, valid_set = get_data(config, log)
    name = str(config.kernel_size) + "_" + str(config.stride) + "_" + config.window_type + "_align.pkl"
    h = load_from_pkl("./hidden_" + name)
    h_v = load_from_pkl("./hidden_v_" + name)
    optimizer, scheduler = get_optimizer(net, config, n_train)
    watcher = LossWatcher(log)
    checker = CheckPoint(config, log)

    for epoch_idx in range(config.max_epoch):
        train_set.batch_shuffle()
        log.log("Batch Shuffled")
        for batch_idx, batch_data in enumerate(train_set):
            start_time = time.time()
            chunk_hidden, chunk_attention_mask, \
            golden_chunks, decoder_inputs, cross_attention, \
            abs_labels, swh_labels, ret_labels, \
            batch_tgt_indices, batch_tgt_indices_mask, batch_src_mask, batch_regular_norm = \
                prepare_data_cuda(batch_data, config, h=h)

            if config.ret_loss_per_token:
                mask = (abs_labels != config.pad).long()
                labels = mask * ret_labels + (1 - mask) * torch.full_like(ret_labels, config.ignore_index)
            else:
                mask = (swh_labels > 0.5).long()
                labels = mask * ret_labels + (1 - mask) * torch.full_like(ret_labels, config.ignore_index)

            # Forward & Backward
            predict = net(
                chunk_hidden=chunk_hidden,
                chunk_attention_mask=chunk_attention_mask,
                input_ids=golden_chunks,
                decoder_input_ids=decoder_inputs,
                encoder_attention_mask=cross_attention,
                src_mask=batch_src_mask,
            )
            n_ret = float(mask.sum())
            n_reg = float(batch_regular_norm.sum())
            loss = loss_func(predict, labels, batch_tgt_indices, batch_tgt_indices_mask, batch_src_mask, n_ret, n_reg).\
                sum()

            backward(loss=loss, optimizer=optimizer, scheduler=scheduler, scaler=scaler if config.amp else None)

            # Loss Watcher
            watcher.append(float(loss))
            watcher.write_log(epoch_idx, batch_idx, time.time() - start_time)

            if checker.ticktock():
                net.eval()
                loss = 0
                total = 0
                with torch.no_grad():
                    for batch_data in valid_set:
                        chunk_hidden, chunk_attention_mask, \
                        golden_chunks, decoder_inputs, cross_attention, \
                        abs_labels, swh_labels, ret_labels, \
                        batch_tgt_indices, batch_tgt_indices_mask, batch_src_mask, batch_regular_norm = \
                            prepare_data_cuda(batch_data, config, h=h_v)

                        if config.ret_loss_per_token:
                            mask = (abs_labels != config.pad).long()
                            labels = mask * ret_labels + (1 - mask) * torch.full_like(ret_labels, config.ignore_index)
                        else:
                            mask = (swh_labels > 0.5).long()
                            labels = mask * ret_labels + (1 - mask) * torch.full_like(ret_labels, config.ignore_index)

                        # Forward & Backward
                        predict = net(
                            chunk_hidden=chunk_hidden,
                            chunk_attention_mask=chunk_attention_mask,
                            input_ids=golden_chunks,
                            decoder_input_ids=decoder_inputs,
                            encoder_attention_mask=cross_attention,
                            src_mask=batch_src_mask,
                        )

                        n_ret = float(mask.sum())
                        n_reg = float(batch_regular_norm.sum())
                        loss += loss_func(predict, labels, batch_tgt_indices, batch_tgt_indices_mask, batch_src_mask,
                                          n_ret, n_reg).sum()
                        total += 1
                if total < 1e-6:
                    loss = 0
                else:
                    loss /= total
                checker.update(loss, net, epoch_idx, batch_idx, config)

        if config.save_each_epoch:
            log.log('Saving Model after %d-th Epoch.' % (epoch_idx + 1))
            save_check_point({
                'epoch': epoch_idx,
                'batch': batch_idx,
                'config': config,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_vloss': 1e99},
                False,
                path=config.save_path,
                file_name='checkpoint_epoch_' + str(epoch_idx) + '.pth.tar'
            )

        log.log('Epoch Finished.')
        gc.collect()
