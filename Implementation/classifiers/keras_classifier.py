import json
import os

import keras

from keras.layers import Dense
from keras.layers import Activation, Input
from keras.layers import BatchNormalization
from keras.layers import Dropout

from keras.models import load_model
from keras.models import save_model
from keras.models import Model

from keras.losses import CategoricalCrossentropy

from keras.optimizers import Adam

import numpy as np

from keras_bert import load_trained_model_from_checkpoint
from transformers import TFBertModel

from Implementation.classifiers.classifier import Classifier
from Implementation.logging.logger import Logger


class KerasCustomClassifier(Classifier):

    def __init__(self, log_path):
        super().__init__()

        self.is_built = False

        self.SAVE_PATH = ""

        self.logger = Logger(log_path=log_path)

        self.bertModel = TFBertModel.from_pretrained(
            "bert-base-uncased",
            num_labels=2,  # binary classification
            output_attentions=False,
            output_hidden_states=False
        )

        self.best_accuracy = 0
        self.data = {"train": dict(), "valid": dict()}
        # self.data = {"train": dict(), "valid": dict(), "test": dict()}

        self.model = []

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
            self.data[data_usecase]["id"] += [int(elem["id"])]
            self.data[data_usecase]["img"] += [elem["img"]]
            self.data[data_usecase]["label"] += [int(elem["label"])]
            self.data[data_usecase]["text"] += [elem["text"]]

    def preprocess(self, args):

        # args example : (BertPreprocessor(pretrained_model_type='bert-base-uncased', do_lower_case=True),
        #                        Img2VecEncoding(precalculated=True, enc_path=ENCODED_IMG_PATH))

        for key, value in self.data.items():

            if type(value) is not dict or \
                    not all(key in value for key in ["id", "label", "text"]) or \
                    not all(len(value["id"]) == len_other
                            for len_other in [len(value["label"]), len(value["text"])]):
                raise Exception("Cannot preprocess \'" + key + "\' data with that format!")

        text_preprocessor = args
            # , image_preprocessor = args

        for key, value in self.data.items():
            value["type"] = key

            text_preprocessor.execute(value)

            # image_preprocessor.execute(value)

    def __build_image_component(self):
        pass

    def build(self):

        # model = load_trained_model_from_checkpoint(
        #     config_file=self.config_path,
        #     checkpoint_file=self.checkpoint_path,
        #     training=False,
        #     trainable=False,
        #     seq_len=self.SEQ_LEN
        # )
        #
        # text_inputs = model.inputs[:2]
        #
        # text_ouputs = model.layers[-3].output

        input_ids = Input(shape=(1, 128), dtype="int32")

        attention_masks = Input(shape=(1, 128), dtype="int32")

        x = self.bertModel([input_ids, attention_masks])

        x = Dense(units=100, kernel_initializer="glorot_uniform")(x)
        x = Activation("relu")(x)

        x = Dense(units=50, kernel_initializer="glorot_uniform")(x)
        x = Activation("relu")(x)

        last_layer = Dense(units=2, activation="relu")(x)

        self.model = Model([input_ids, attention_masks], last_layer)

        optimizer = Adam(learning_rate=0.001)
        loss = CategoricalCrossentropy()
        metrics = ['accuracy']

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, args):

        self.batch_size, epochs, self.learning_rate, self.SAVE_PATH = args

        if not self.is_built:
            self.build()





    # def compute_accuracy(self, outputs, labels):
    #
    #     actual = []
    #
    #     for out in outputs:
    #         if out[0] > out[1]:
    #             actual += [0]
    #         else:
    #             actual += [1]
    #
    #     hits = 0
    #     for index in range(0, len(actual)):
    #         if actual[index] == labels[index]:
    #             hits += 1
    #
    #     return hits

    def test_model(self, data_used="test"):
        pass
        # dataloader = DataLoader(dataset=self.dataset_objs[data_used], batch_size=self.batch_size, shuffle=True)
        #
        # accuracy = 0
        #
        # # Enables eval mode
        # self.model.eval()
        #
        # loss = 0
        # crit = nn.CrossEntropyLoss()
        #
        # # Don't update the gradients
        # with torch.no_grad():
        #     for batch_index, (inputs, labels) in enumerate(dataloader):
        #         dev_inputs = inputs.to(DEVICE)
        #         dev_labels = labels.to(DEVICE)
        #
        #         # forward
        #         outputs = self.model(dev_inputs)
        #
        #         loss += crit(outputs, dev_labels).item()
        #
        #         accuracy += self.compute_accuracy(outputs, dev_labels)
        #
        # # Go back to training mode
        # self.model.train()
        #
        # accuracy /= len(self.data[data_used]["label"])
        # loss /= len(self.data[data_used]["label"])
        #
        # return accuracy, loss

    def save_model(self, path, ext):
        save_model(self.model, path + ext)

    def load_model(self, load_path):
        self.model = load_model(load_path)

    def show_best_result(self, args) -> float:
        return self.best_accuracy
