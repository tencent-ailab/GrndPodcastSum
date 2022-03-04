from copy import deepcopy as cp
import torch

from utility import (
    prepare_test,
    prepare_ext_test,
    prepare_abs_test,
    convert_cuda,
    prepare_abs_emb_test,
    pick_golden_chunks,
    do_tricks
)


def beam_search(net, batch_data, config, h=None, output_others=False):
    with torch.no_grad():
        batch_chunks, batch_indices = convert_cuda(prepare_test(batch_data, config), config)
        ref_chunk_ids = pick_golden_chunks(batch_chunks, batch_data[1])[0]

        """
            Get Salience Score
        """
        net.set_mode("ext")
        # extractor parts: chunk_hidden, chunk_attention_mask
        chunk_hidden, chunk_attention_mask = convert_cuda(prepare_ext_test(batch_indices, config, h), config)
        salience = net(
            chunk_hidden=chunk_hidden,
            chunk_attention_mask=chunk_attention_mask,
        )
        # Beam Search Decoding

        golden_chunk_ids = [0]
        summary = [config.cls]
        summary_sents = [[config.cls]]

        start_state = [0, summary, summary_sents, golden_chunk_ids]
        candidates = [[start_state]]

        flag = False
        answers = []

        for __ in range(config.gen_max_len):
            cands = candidates[-1]
            next_cands = []

            for cand in cands:
                score, summary, summary_sents, golden_chunks = cand

                # Generate Next Token
                net.set_mode("abs")
                encoder_input_ids, decoder_input_ids, cross_attention_mask = \
                    convert_cuda(prepare_abs_test(batch_chunks, golden_chunks, summary_sents, config), config)

                predict = net(
                    input_ids=encoder_input_ids,
                    decoder_input_ids=decoder_input_ids,
                    encoder_attention_mask=cross_attention_mask,
                )[:, -1, :]

                # predict = net.adjust_logits_during_generation(
                #    predict, cur_len=len(summary), max_length=config.gen_max_len
                # )[0]

                predict = torch.log_softmax(predict, dim=-1)[0]
                predict = do_tricks(predict, score, summary, config)

                next_probs, next_tokens = torch.topk(predict, k=config.beam_size)

                for next_prob, next_token in zip(next_probs.tolist(), next_tokens.tolist()):
                    # Predict if switch or not
                    net.set_mode("swh")
                    shift_token = convert_cuda(prepare_abs_emb_test(summary_sents, int(next_token), config), config)
                    predict_swh = torch.sigmoid(net(
                        input_ids=encoder_input_ids,
                        decoder_input_ids=decoder_input_ids,
                        encoder_attention_mask=cross_attention_mask,
                        decoder_labels=shift_token,
                    ))
                    prob_swh = float(predict_swh[0, -1])
                    switch = (prob_swh >= config.Th)

                    new_summary_sents = cp(summary_sents)
                    new_golden_chunks = cp(golden_chunks)

                    if switch:
                        # Predict Next Chunk, top 1 chunk only
                        net.set_mode("ret")
                        predict_ret = torch.log_softmax(net(
                            chunk_hidden=chunk_hidden,
                            chunk_attention_mask=chunk_attention_mask,
                            salience=salience,
                            input_ids=encoder_input_ids,
                            decoder_input_ids=decoder_input_ids,
                            encoder_attention_mask=cross_attention_mask
                        ), dim=-1)

                        next_chunk = int(torch.argmax(predict_ret[0, -1]))
                        new_golden_chunks.append(next_chunk)
                        new_summary_sents.append([next_token])
                    else:
                        new_summary_sents[-1].append(next_token)

                    new_score = score + float(next_prob)
                    new_summary = cp(summary) + [next_token]
                    new_state = [new_score, new_summary, new_summary_sents, new_golden_chunks]
                    next_cands.append(new_state)

            next_cands = sorted(next_cands, key=lambda x: -x[0] / (len(x[1]) ** config.length_penalty))
            if len(next_cands) > config.beam_size * 2:
                next_cands = next_cands[:config.beam_size * 2]
            for_update = []

            for rank, new_cand in enumerate(next_cands):
                new_score, new_summary, new_summary_sents, new_golden_chunks = new_cand
                if new_summary[-1] == config.sep:
                    if rank > config.beam_size:
                        continue
                    if len(new_summary) < config.gen_min_len:
                        continue
                    answers.append([new_score, new_summary, new_summary_sents, new_golden_chunks])
                    if len(answers) >= config.beam_size:
                        flag = True
                else:
                    for_update.append([new_score, new_summary, new_summary_sents, new_golden_chunks])
            # collect enough answer
            if flag:
                break

            if len(for_update) > config.beam_size:
                for_update = for_update[:config.beam_size]

            candidates.append(for_update)

        if len(answers) < 1:
            answers = candidates[-1]

        answers = sorted(answers, key=lambda x: -x[0] / (len(x[1]) ** config.length_penalty))

        if not output_others:
            return answers[0]

        return answers[0], ref_chunk_ids
