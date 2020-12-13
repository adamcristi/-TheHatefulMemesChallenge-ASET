import tensorflow
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel, BertTokenizerFast

from Implementation.preprocess.preprocessor import Preprocessor

import numpy as np


class BertPreprocessor(Preprocessor):

    def __init__(self, pretrained_model_type, do_lower_case, load_bert=False):
        super().__init__()

        self.pretrained_model_type = pretrained_model_type
        self.do_lower_case = do_lower_case
        self.load_bert = load_bert

        # get bert model
        self.bert_model = TFBertModel.from_pretrained(self.pretrained_model_type)

        # get tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_model_type, do_lower_case=self.do_lower_case)

    def execute(self, data):

        if self.load_bert:
            data["bert_output"] = np.load("./bert_output/bert_output_" + data["type"] + ".npy")

            # print(data["bert_output"][0])
            # print(len(data["bert_output"]))
            # print(len(data["bert_output"][0]))
            # print(data["bert_output"][0].shape)

        else:
            input_ids = []
            attention_masks = []

            # tokenize all of the texts
            for text in data["text"]:
                encoded = self.tokenizer.encode_plus(
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
                    return_attention_mask=True,
                )

                # add the encoded text
                input_ids.append(tensorflow.reshape(encoded['input_ids'], (1, -1)))

                # add the attention mask
                attention_masks.append(tensorflow.reshape(encoded['attention_mask'], (1, -1)))

            print("\n\nComputing BERT output - " + data["type"] + " ...\n")
            bert_output = np.array([self.bert_model(input_id, token_type_ids=None, attention_mask=attention_mask)[1]
                                    for input_id, attention_mask in tqdm(zip(input_ids, attention_masks))])

            np.save("./bert_output/bert_output_" + data["type"] + "_new.npy", bert_output)
            data["bert_output"] = bert_output
