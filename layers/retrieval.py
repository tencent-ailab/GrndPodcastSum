import math
import torch
import torch.nn as nn
from .classification_head import BinaryClsHead
from .extractor import Extractor
from .abstracter import Abstracter


class CrossAttention(nn.Module):
    """
        Attention Score = q^T W k + v^T tanh(W'[q; k] + b)
                       ~= (q^T U) (V^T k) + v^T tanh(W'[q; k] + b)

        Type of Attention:
            1: linear
            2: bilinear
            3: linear + bilinear
    """

    def __init__(self, config):
        super().__init__()
        self.type_att = config.type_att
        n_query = config.d_query
        n_key = config.d_key
        n_out = config.d_att

        self.bi_linear_query = nn.Linear(n_query, n_out)
        self.bi_linear_key = nn.Linear(n_key, n_out)

        self.linear_query = nn.Linear(n_query, n_out)
        self.linear_key = nn.Linear(n_key, n_out)
        self.att = nn.Linear(n_out, 1)

    def low_rank_bi_linear(self, query, key):
        """
            Relevancy Score
            Low Rank Bi-Linear Attention Implementation

            query: bsz, num_q, dim_q
            key: bsz, num_k, dim_k
            att_logits: bsz, num_q, num_k

            att_logits = q^T W k ~= q^T (U V^T) k = (q^T U) (V^T k)
        """
        query = self.bi_linear_query(query)
        # bsz, num_q, dim
        key = self.bi_linear_key(key)
        # bsz, num_k, dim

        d_k = query.shape[-1]
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        return logits

    def linear(self, query, key):
        """
            Saliency Score
            Linear Attention Implementation

            query: bsz, num_q, dim_q
            key: bsz, num_k, dim_k
            att_logits: bsz, num_q, num_k

            att_logits = v^T tanh(W'[q; k] + b)
        """
        query = self.linear_query(query).permute(0, 2, 1).unsqueeze(-1)
        key = self.linear_key(key).permute(0, 2, 1).unsqueeze(-2)
        activation = (query + key).permute(0, 2, 3, 1)
        hidden = torch.tanh(activation)
        logits = self.att(hidden)
        return logits.squeeze(-1)

    def forward(self, query, key):
        ret = 0
        if self.type_att & 1 > 0:
            a = self.linear(query, key)
            ret += a
        if self.type_att & 2 > 0:
            b = self.low_rank_bi_linear(query, key)
            ret += b
        return ret


class Retrieval(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mode = config.mode
        self.extractor = Extractor(config)
        self.abstracter = Abstracter.from_pretrained(config.model_abs)
        self.switch = BinaryClsHead(config.d_model * 2, config.d_inner, config.dropout)
        self.retrieval = CrossAttention(config)
        self.epsilon = config.epsilon

    def set_mode(self, mode):
        """
            mode: str
                ext: only extractor parameters
                abs: only abstractor parameters
                ret: retrieval parameters
                swh: switch parameters
        """
        self.mode = mode
        for para in self.parameters():
            para.requires_grad = False

        if self.mode == "ext":
            for para in self.extractor.parameters():
                para.requires_grad = True
        elif self.mode == "abs":
            for para in self.abstracter.parameters():
                para.requires_grad = True
        elif self.mode == "ret":
            for para in self.retrieval.parameters():
                para.requires_grad = True
        elif self.mode == "swh":
            for para in self.switch.parameters():
                para.requires_grad = True

    def get_mode(self):
        return self.mode

    def retrieve(
            self,
            # Extractor
            chunk_input_ids=None,
            chunk_hidden=None,
            chunk_attention_mask=None,
            salience=None,
            # Abstracter
            hidden=None,
            src_mask=None,
    ):
        if salience is None:
            with torch.no_grad():
                # bsz, n_key
                salience = self.extractor(
                    chunk_input_ids=chunk_input_ids,
                    chunk_hidden=chunk_hidden,
                    chunk_attention_mask=chunk_attention_mask
                )
        if src_mask is None:
            src_mask = torch.ones_like(salience)

        src_mask_ = (src_mask.unsqueeze(1)-1) * 1e6
        relevance = self.retrieval(hidden, chunk_hidden)
        attention = torch.log_softmax(relevance + self.epsilon * salience.unsqueeze(1) + src_mask_, dim=-1)
        return attention

    def forward(
            self,
            # Extractor
            chunk_input_ids=None,
            chunk_hidden=None,
            chunk_attention_mask=None,
            # Abstracter
            input_ids=None,
            decoder_input_ids=None,
            encoder_attention_mask=None,
            # Decoding
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            # Switch
            decoder_labels=None,
            # Retrieval
            src_mask=None,
            salience=None,
    ):
        if self.mode == "ext":
            return self.extractor(
                chunk_input_ids=chunk_input_ids,
                chunk_hidden=chunk_hidden,
                chunk_attention_mask=chunk_attention_mask
            )
        elif self.mode == "abs":
            return self.abstracter(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                hidden_only=False
            )
        else:
            with torch.no_grad():
                hidden = self.abstracter(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    hidden_only=True
                )
            if self.mode == 'ret':
                return self.retrieve(
                    chunk_input_ids=chunk_input_ids,
                    chunk_hidden=chunk_hidden,
                    chunk_attention_mask=chunk_attention_mask,
                    salience=salience,
                    hidden=hidden,
                    src_mask=src_mask,
                )
            elif self.mode == "swh":
                next_embedding = self.abstracter.model.shared(decoder_labels) * self.abstracter.model.embed_scale
                return self.switch(torch.cat([hidden, next_embedding], dim=-1))

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(logits, self.config.cls)
        if cur_len == max_length - 1 and self.config.sep is not None:
            self._force_token_ids_generation(logits, self.config.sep)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.n_vocab) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")
