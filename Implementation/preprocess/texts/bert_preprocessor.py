import torch
from transformers import BertTokenizer

from Implementation.preprocess.preprocessor import Preprocessor


class BertPreprocessor(Preprocessor):

    def __init__(self, pretrained_model_type, do_lower_case):

        super().__init__()

        self.pretrained_model_type = pretrained_model_type
        self.do_lower_case = do_lower_case

    def execute(self, data):

        # get tokenizer
        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_type, do_lower_case=self.do_lower_case)

        # find the maximum token length
        max_len = 0
        for text in data["text"]:
            max_len = max(max_len, len(tokenizer.encode(text, add_special_tokens=True)))

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
                max_length=64,
                # return pytorch tensors
                return_tensors='pt',
                return_attention_mask=True,
            )

            # add the encoded text
            input_.append(encoded['input_ids'])

            # add the attention mask
            attention_masks.append(encoded['attention_mask'])

        # convert lists to tensors
        data["attention_masks"] = torch.cat(attention_masks, dim=0)
        data["encoded_text"] = torch.cat(input_, dim=0)
        data["label"] = torch.tensor(data["label"])
