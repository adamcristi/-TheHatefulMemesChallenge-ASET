import json
import os
import time
import cv2
import keras

from keras.layers import Dense
from keras.layers import Activation, Input, GlobalMaxPooling2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, \
    Add
from keras.layers import BatchNormalization
from keras.layers import Dropout

from keras.models import load_model
from keras.models import save_model
from keras.models import Model

from keras.losses import BinaryCrossentropy

from keras.optimizers import Adam

import numpy as np

# from keras_bert import load_trained_model_from_checkpoint
# from transformers import TFBertModel
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

from PIL import Image

from Implementation.classifiers.classifier import Classifier
from Implementation.logging.logger import Logger


class KerasCustomClassifier(Classifier):

    def __init__(self, log_path):
        super().__init__()

        self.is_built = False

        self.IMAGE_COMPLETE_PATH = "./data/data/"
        self.SAVE_PATH = ""

        self.logger = Logger(log_path=log_path)

        self.best_accuracy = 0
        self.data = {"train": dict(), "valid": dict()}
        # self.data = {"train": dict(), "valid": dict(), "test": dict()}

        self.model = Model()

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

        self.data[data_usecase]["label"] = np.array([self.data[data_usecase]["label"]]).transpose()
        # self.data[data_usecase]["label"] = np.array([np.asarray(self.data[data_usecase]["label"]).tolist()])

    def preprocess(self, text_preprocessor):

        # args example : (BertPreprocessor(pretrained_model_type='bert-base-uncased', do_lower_case=True,
        #                                                                               load_bert=False))

        # for key, value in self.data.items():
        #
        #     if type(value) is not dict or \
        #             not all(key in value for key in ["id", "label", "text"]) or \
        #             not all(len(value["id"]) == len_other
        #                     for len_other in [len(value["label"]), len(value["text"])]):
        #         raise Exception("Cannot preprocess \'" + key + "\' data with that format!")

        for key, value in self.data.items():
            value["type"] = key

            text_preprocessor.execute(value)

            value["image_data"] = np.array([
                cv2.resize(cv2.cvtColor(cv2.imread(filename=self.IMAGE_COMPLETE_PATH + img_path), cv2.COLOR_BGR2RGB),
                           (32, 32)).astype('float32')
                for img_path in tqdm(value["img"])])

            value["model_input"] = [value["bert_output"][:value["image_data"].shape[0]], value["image_data"]]

            # print(value["bert_output"].tolist())
            # print(type(value["model_input"][0][1][0]))
            # print(len(value["model_input"][0][0]))
            # print(len(value["model_input"][0][1]))
            # print(len(value["model_input"][0][2]))
            # print(type(value["model_input"][1][0][0][0][0]))

            # print(value["model_input"][0])
            # print(value["model_input"][0, 0].shape)
            # print(value["model_input"][1, 0].shape)

    @staticmethod
    def __build_image_component():

        input_ = Input(shape=(32, 32, 3))
        # this is for variable input (it should work, but for speed I opted for destructive resize in opencv)
        # input_ = Input(shape=(None, None, 3))

        x = Conv2D(32, (3, 3), activation="relu")(input_)
        x = BatchNormalization()(x)

        x = Conv2D(32, (3, 3), activation="relu")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Dropout(0.5)(x)

        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)

        output_ = Dense(768, activation='relu')(x)

        return input_, output_

    def build(self, *args):

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

        image_input, image_output = self.__build_image_component()

        text_input = Input(shape=(1, 768), dtype="float32")

        x = Add()([text_input, image_output])
        x = BatchNormalization()(x)

        x = Dense(units=256, kernel_initializer="glorot_normal")(x)
        x = Activation("relu")(x)
        x = Dropout(0.25)(x)

        x = Dense(units=128, kernel_initializer="glorot_normal")(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)

        last_layer = Dense(units=2, activation="relu")(x)

        self.model = Model([text_input, image_input], last_layer)
        # self.model = Model(image_input, image_output)

        print(self.model.summary())

        optimizer = Adam(learning_rate=self.learning_rate)
        loss = BinaryCrossentropy()
        metrics = ['accuracy']

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, *args):

        self.batch_size, epochs, self.learning_rate, self.SAVE_PATH = args

        if not self.is_built:
            self.build()

        mcp = ModelCheckpoint(self.SAVE_PATH + ".h5",
                              save_best_only=True,
                              monitor="val_accuracy",
                              mode="max")

        print(self.data["train"]["model_input"][0].shape)
        print(self.data["train"]["model_input"][1].shape)
        print(self.data["train"]["label"].shape)

        self.model.fit(x=self.data["train"]["model_input"],
                       y=self.data["train"]["label"],
                       validation_data=(self.data["valid"]["model_input"], self.data["valid"]["label"]),
                       epochs=epochs,
                       batch_size=self.batch_size,
                       verbose=2,
                       callbacks=[mcp],
                       shuffle=True)

    def test_model(self, data_used="test"):
        return

    def save_model(self, path, ext):
        save_model(self.model, path + ext)

    def load_model(self, load_path):
        self.model = load_model(load_path)

    def predict(self):
        return

    def show_best_result(self, args) -> float:
        return self.best_accuracy
