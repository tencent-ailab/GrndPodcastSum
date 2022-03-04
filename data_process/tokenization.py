from transformers import BertTokenizer, GPT2Tokenizer
from utility import detokenize

Tokenizer_Mapping = {
    "bert-base-uncased": BertTokenizer,
    "bert-base-cased": BertTokenizer,
    "bert-large-uncased": BertTokenizer,
    "bert-large-cased": BertTokenizer,
    "roberta-base": GPT2Tokenizer,
    "roberta-large": GPT2Tokenizer,
    "facebook/bart-base": GPT2Tokenizer,
    "facebook/bart-large": GPT2Tokenizer,
    "facebook/bart-large-cnn": GPT2Tokenizer,
    "../../pretrained_models/bert-base-uncased": BertTokenizer,
    "../../pretrained_models/bert-base-cased": BertTokenizer,
    "../../pretrained_models/bert-large-uncased": BertTokenizer,
    "../../pretrained_models/bert-large-cased": BertTokenizer,
    "../../pretrained_models/roberta-base": GPT2Tokenizer,
    "../../pretrained_models/roberta-large": GPT2Tokenizer,
    "../../pretrained_models/facebook/bart-base": GPT2Tokenizer,
    "../../pretrained_models/facebook/bart-large": GPT2Tokenizer,
    "../../pretrained_models/facebook/bart-large-cnn": GPT2Tokenizer
}

Encode_Mapping = {
    "bert-base-uncased": "Bert",
    "bert-base-cased": "Bert",
    "bert-large-uncased": "Bert",
    "bert-large-cased": "Bert",
    "roberta-base": "GPT",
    "roberta-large": "GPT",
    "facebook/bart-base": "GPT",
    "facebook/bart-large": "GPT",
    "facebook/bart-large-cnn": "GPT",
    "../../pretrained_models/bert-base-uncased": "Bert",
    "../../pretrained_models/bert-base-cased": "Bert",
    "../../pretrained_models/bert-large-uncased": "Bert",
    "../../pretrained_models/bert-large-cased": "Bert",
    "../../pretrained_models/roberta-base": "GPT",
    "../../pretrained_models/roberta-large": "GPT",
    "../../pretrained_models/facebook/bart-base": "GPT",
    "../../pretrained_models/facebook/bart-large": "GPT",
    "../../pretrained_models/facebook/bart-large-cnn": "GPT"
}


class MyTokenizer:
    def __init__(self, config):
        self.sep = config.sep
        self.cls = config.cls
        self.model = config.model_ext
        self.tokenizer = None
        if self.model in Tokenizer_Mapping:
            self.tokenizer = Tokenizer_Mapping[self.model].from_pretrained(self.model)

    def encode(self, sequence):
        if Encode_Mapping[self.model] == "Bert":
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sequence))
        elif Encode_Mapping[self.model] == "GPT":
            return self.tokenizer.encode(sequence, add_special_tokens=False)

    def decode(self, sequence, mapping=None):
        cut_off = []
        for token in sequence:
            if token == self.sep:
                break
            elif token != self.cls:
                cut_off.append(token)

        if len(cut_off) == 0:
            return ""

        if Encode_Mapping[self.model] == "Bert":
            tokens = self.tokenizer.convert_ids_to_tokens(cut_off)
            tokens = [token.replace('##', '') if token.startswith('##') else ' ' + token for token in tokens]
            return detokenize("".join(tokens)[1:], mapping)
        elif Encode_Mapping[self.model] == "GPT":
            return self.tokenizer.decode(cut_off, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def tokenize(self, sequence):
        return self.tokenizer.tokenize(sequence)
