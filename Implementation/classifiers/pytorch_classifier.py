import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel

from Implementation.classifiers.classifier import *
from Implementation.logging.logger import Logger
from Implementation.preprocess.images.img2vecencoding import Img2VecEncoding
from Implementation.preprocess.texts.bert_preprocessor_torch import BertPreprocessor

import numpy as np
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocalNeuralNet(nn.Module):
    def __init__(self, input_size):
        super(LocalNeuralNet, self).__init__()

        first_layer = 256
        second_layer = 128

        # first hidden layer
        self.linear_layer_1 = nn.Linear(input_size, first_layer)
        self.activation_1 = nn.ReLU()
        self.batchnorm_1 = nn.BatchNorm1d(first_layer, affine=False)
        self.dropout_1 = nn.Dropout(p=0.2)

        # second hidden layer
        self.linear_layer_2 = nn.Linear(first_layer, second_layer)
        self.activation_2 = nn.ReLU()
        self.batchnorm_2 = nn.BatchNorm1d(second_layer, affine=False)
        self.dropout_2 = nn.Dropout(p=0.4)

        # last layer
        self.linear_last = nn.Linear(second_layer, 2)
        # we use cross-entropy and it applies softmax automatically, so we don't need softmax here

    def forward(self, input_):
        output = self.linear_layer_1(input_)
        output = self.activation_1(output)
        #output = self.batchnorm_1(output)
        output = self.dropout_1(output)

        output = self.linear_layer_2(output)
        output = self.activation_2(output)
        #output = self.batchnorm_2(output)
        output = self.dropout_2(output)

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


class PytorchCustomClassifier(Classifier):

    def __init__(self, log_path):
        super().__init__()

        self.SAVE_PATH = ""

        self.logger = Logger(log_path=log_path)

        self.bertModel = BertModel.from_pretrained(
            "bert-base-uncased",
            num_labels=2,  # binary classification
            output_attentions=False,
            output_hidden_states=False
        ).requires_grad_(False).to(DEVICE)

        self.best_accuracy = 0
        self.data = {"train": dict(), "valid": dict()}
        # self.data = {"train": dict(), "valid": dict(), "test": dict()}

        self.model = []

        self.dataset_objs = {
            "train": HatefulMemesDataset(self.data["train"], self.bertModel),
            "valid": HatefulMemesDataset(self.data["valid"], self.bertModel),
            # "test": HatefulMemesDataset(self.data["test"], self.bertModel)
        }

        self.batch_size = 1
        self.learning_rate = 0.1

    def load_data(self, args):
        # dataset_usecase can be train, valid or test
        data_file, data_usecase = args
        temp_data = []

        try:
            with open(data_file, "r") as file:
                for line in file:
                    temp_data += [json.loads(line)]

        except Exception as e:
            raise

        self.data[data_usecase] = dict({"id": [], "img": [], "label": [], "text": []})

        for elem in temp_data:
            self.data[data_usecase]["id"] += [torch.tensor(int(elem["id"]))]
            self.data[data_usecase]["img"] += [elem["img"]]
            self.data[data_usecase]["label"] += [torch.tensor(int(elem["label"]))]
            self.data[data_usecase]["text"] += [elem["text"]]

        self.data[data_usecase]["id"] = torch.tensor(self.data[data_usecase]["id"])
        self.data[data_usecase]["label"] = torch.tensor(self.data[data_usecase]["label"])

    def preprocess(self, args):

        # args example : (BertPreprocessor(pretrained_model_type='bert-base-uncased', do_lower_case=True),
        #                        Img2VecEncoding(precalculated=True, enc_path=ENCODED_IMG_PATH))

        for key, value in self.data.items():

            if type(value) is not dict or \
                    not all(key in value for key in ["id", "label", "text"]) or \
                    not all(len(value["id"]) == len_other
                            for len_other in [len(value["label"]), len(value["text"])]):
                raise Exception("Cannot preprocess \'" + key + "\' data with that format!")

        text_preprocessor, image_preprocessor = args

        for key, value in self.data.items():

            value["type"] = key

            text_preprocessor.execute(value)

            image_preprocessor.execute(value)

            self.dataset_objs[key] = HatefulMemesDataset(value, self.bertModel)

    def train(self, args):

        self.batch_size, epochs, self.learning_rate, self.SAVE_PATH = args

        train_dataset = self.dataset_objs["train"]

        iterations = np.math.ceil(len(train_dataset) / self.batch_size)

        dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        # the size of the input (bertModel+imgEncoded)
        input_size = next(iter(train_dataset))[0].shape[0]

        # the model to train
        self.model = LocalNeuralNet(input_size).requires_grad_(True).to(DEVICE)

        # Enables training mode
        self.model.train()

        # loss criterion
        crit = nn.CrossEntropyLoss()

        # optimizer
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)

        # train loop
        with torch.enable_grad():
            for epoch in tqdm(range(0, epochs), position=0):
                for batch_index, (inputs, labels) in tqdm(enumerate(dataloader), position=1):
                    dev_inputs = inputs.to(DEVICE)
                    dev_labels = labels.to(DEVICE)

                    # forward
                    outputs = self.model(dev_inputs)

                    loss = crit(outputs, dev_labels)

                    # backward
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                train_accuracy, train_loss = self.test_model(data_used="train")
                valid_accuracy, valid_loss = self.test_model(data_used="valid")

                if valid_accuracy > self.best_accuracy:
                    self.best_accuracy = valid_accuracy

                print(f"Epoch {epoch + 1}/{epochs}, train_loss: {train_loss}, train_accuracy: {train_accuracy}, " +
                      f"valid_loss: {valid_loss}, valid_accuracy: {valid_accuracy}")

                self.logger.log({"epoch": str(epoch + 1) + "/" + str(epochs),
                                 "train_loss": train_loss,
                                 "train_accuracy": train_accuracy,
                                 "valid_loss": valid_loss,
                                 "valid_accuracy": valid_accuracy})

                if self.SAVE_PATH != "":
                    self.save_model(self.SAVE_PATH + str(time.time()) + "_epoch_" + str(epoch + 1) + "_" + str(epochs),
                                    ".pth")

        # if self.SAVE_PATH != "":
        #     self.save_model(self.SAVE_PATH + str(time.time()) + "_final", ".pth")

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

    def test_model(self, data_used="test"):

        dataloader = DataLoader(dataset=self.dataset_objs[data_used], batch_size=self.batch_size, shuffle=True)

        accuracy = 0

        # Enables eval mode
        self.model.eval()

        loss = 0
        crit = nn.CrossEntropyLoss()

        # Don't update the gradients
        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(dataloader):
                dev_inputs = inputs.to(DEVICE)
                dev_labels = labels.to(DEVICE)

                # forward
                outputs = self.model(dev_inputs)

                loss += crit(outputs, dev_labels).item()

                accuracy += self.compute_accuracy(outputs, dev_labels)

        # Go back to training mode
        self.model.train()

        accuracy /= len(self.data[data_used]["label"])
        loss /= len(self.data[data_used]["label"])

        return accuracy, loss

    def save_model(self, path, ext):
        torch.save(self.model, path+ext)

    def load_model(self, load_path):
        self.model = torch.load(load_path)

    def show_best_result(self, args) -> float:
        return self.best_accuracy
