import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel

from Implementation.classifiers.classifier import *
from Implementation.preprocess.images.img2vecencoding import Img2VecEncoding
from Implementation.preprocess.texts.bert_preprocessor import BertPreprocessor

import numpy as np
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocalNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LocalNeuralNet, self).__init__()

        self.linear_layer_1 = nn.Linear(input_size, hidden_size)
        self.f1 = nn.Sigmoid()

        self.linear_last = nn.Linear(hidden_size, 2)
        # we use cross-entropy and it applies softmax automatically, so we don't need softmax here

    def forward(self, input_):
        output = self.linear_layer_1(input_)
        output = self.f1(output)

        output = self.linear_last(output)

        return output


class HatefulMemesDataset(Dataset):

    def __init__(self, data, bertModel):
        self.data = data
        self.bertModel = bertModel

    def __getitem__(self, index):
        input_ = self.data["encoded_text"][index]
        attention_mask = self.data["attention_masks"][index]

        # send the two inputs to the GPU (only if cuda is available, see DEVICE variable above)
        dev_input_ = input_.to(DEVICE)
        dev_attention_mask = attention_mask.to(DEVICE)

        # feedforward through BERT model
        loss, logits = self.bertModel.forward(input_ids=dev_input_,
                                              token_type_ids=None,
                                              attention_mask=dev_attention_mask)

        dev_encoded_image = self.data["img_encoded"][index].to(DEVICE)

        # return bertOutput + imgEncoded
        return torch.cat((logits, dev_encoded_image), dim=1)[0], self.data["label"][index]

    def __len__(self):
        return len(self.data["encoded_text"])


class CustomClassifier(Classifier):

    def __init__(self):
        super().__init__()

        self.SAVE_PATH = ""

        self.bertModel = BertModel.from_pretrained(
            "bert-base-uncased",
            num_labels=2,  # binary classification
            output_attentions=False,
            output_hidden_states=False
        ).requires_grad_(False).to(DEVICE)

        self.best_accuracy = 0
        self.data = dict()

        self.model = []

        self.dataset_obj = HatefulMemesDataset(self.data, self.bertModel)

        self.batch_size = 1
        self.learning_rate = 0.1

    def load_data(self, args):
        data_file = args
        temp_data = []

        try:
            with open(data_file, "r") as file:
                for line in file:
                    temp_data += [json.loads(line)]

        except Exception as e:
            raise

        self.data = dict({"id": [], "img": [], "label": [], "text": []})
        for elem in temp_data:
            self.data["id"] += [torch.tensor(int(elem["id"]))]
            self.data["img"] += [elem["img"]]
            self.data["label"] += [torch.tensor(int(elem["label"]))]
            self.data["text"] += [elem["text"]]

        self.data["id"] = torch.tensor(self.data["id"])
        self.data["label"] = torch.tensor(self.data["label"])

    def preprocess(self, args):

        # args example : (BertPreprocessor(pretrained_model_type='bert-base-uncased', do_lower_case=True),
        #                        Img2VecEncoding(precalculated=True, enc_path=ENCODED_IMG_PATH))

        if type(self.data) is not dict or \
                not all(key in self.data for key in ["id", "label", "text"]) or \
                not all(len(self.data["id"]) == len_other
                        for len_other in [len(self.data["label"]), len(self.data["text"])]):
            raise Exception("Cannot preprocess data with that format!")

        text_preprocessor, image_preprocessor = args

        text_preprocessor.execute(self.data)

        image_preprocessor.execute(self.data)

        self.dataset_obj = HatefulMemesDataset(self.data, self.bertModel)

    def train(self, args):

        self.batch_size, epochs, self.learning_rate, self.SAVE_PATH = args

        iterations = np.math.ceil(len(self.dataset_obj) / self.batch_size)

        dataloader = DataLoader(dataset=self.dataset_obj, batch_size=self.batch_size, shuffle=True)

        # the size of the input (bertModel+imgEncoded)
        input_size = next(iter(self.dataset_obj))[0].shape[0]

        # the model to train
        self.model = LocalNeuralNet(input_size, 256).requires_grad_(True).to(DEVICE)

        # Enables training mode
        self.model.train()

        # loss criterion
        crit = nn.CrossEntropyLoss()

        # optimizer
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # train loop
        with torch.enable_grad():
            for epoch in tqdm(range(0, epochs)):
                for batch_index, (inputs, labels) in enumerate(dataloader):
                    dev_inputs = inputs.to(DEVICE)
                    dev_labels = labels.to(DEVICE)

                    # forward
                    outputs = self.model(dev_inputs)

                    loss = crit(outputs, dev_labels)

                    # backward
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    # if (batch_index + 1) % 10 == 0:
                    #     print(f"Epoch {epoch + 1}/{epochs}, Step {batch_index + 1}/{iterations}, inputs {inputs.shape}")

                print(f"Epoch loss: {loss.item()}")
                self.test_model()

        if self.SAVE_PATH != "":
            self.save_model()

    def compute_accuracy(self, outputs, labels):

        actual = []

        for out in outputs:
            if out[0] > out[1]:
                actual += [0]
            else:
                actual += [1]

        hits = 0
        for index in range(0, len(actual)):
            if actual[index] == labels[index]:
                hits += 1

        return hits

    def test_model(self):

        dataloader = DataLoader(dataset=self.dataset_obj, batch_size=self.batch_size, shuffle=True)

        accuracy = 0

        # Enables eval mode
        self.model.eval()

        # Don't update the gradients
        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(dataloader):
                dev_inputs = inputs.to(DEVICE)
                dev_labels = labels.to(DEVICE)

                # forward
                outputs = self.model(dev_inputs)

                accuracy += self.compute_accuracy(outputs, dev_labels)

        # Go back to training mode
        self.model.train()

        accuracy /= len(self.data["label"])

        print(accuracy)
        return accuracy

    def save_model(self):
        torch.save(self.model, self.SAVE_PATH)

    def load_model(self, load_path):
        self.model = torch.load(load_path)

    def show_best_result(self, args) -> float:
        return self.best_accuracy
