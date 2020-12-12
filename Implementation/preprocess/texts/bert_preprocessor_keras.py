import os

import tensorflow
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from transformers import BertTokenizer

from Implementation.preprocess.preprocessor import Preprocessor

import numpy as np


class BertPreprocessor(Preprocessor):

    def __init__(self, pretrained_model_type, do_lower_case):

        super().__init__()

        self.pretrained_model_type = pretrained_model_type
        self.do_lower_case = do_lower_case

    def execute(self, data):

        # get tokenizer
        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_type, do_lower_case=self.do_lower_case)

        attention_masks = []
        input_ = []

        # tokenize all of the texts
        for text in data["text"]:
            encoded = tokenizer.encode_plus(
                # text to encode
                text,
                # adds "[CLS]" and "[SEP]" tokens
                add_special_tokens=True,
                # pad and truncate all texts
                padding="max_length",
                truncation=True,
                max_length=128,
                # return tensorflow tensors
                return_tensors='tf',
                return_attention_mask=True
            )

            # add the encoded text
            input_.append(encoded['input_ids'])

            # add the attention mask
            attention_masks.append(encoded['attention_mask'])

        data["attention_masks"] = np.array(attention_masks)
        data["encoded_text"] = np.array(input_)
