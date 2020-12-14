import json
import os
import time
import cv2

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation, Input, GlobalMaxPooling2D, Conv2D, MaxPooling2D, \
    GlobalAveragePooling2D, Flatten, Add, Lambda, Concatenate, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Model

from tensorflow.keras.regularizers import l2

from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import ResNet50, resnet50

import numpy as np

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

import tensorflow as tf

from Implementation.classifiers.classifier import Classifier
from Implementation.logging.logger import Logger


class KerasCustomClassifier(Classifier):

    def __init__(self, log_path):
        super().__init__()

        self.is_built = False

        self.image_res = (256, 256, 3)

        self.IMAGE_COMPLETE_PATH = "./data/data/"
        self.SAVE_PATH = ""

        self.logger = Logger(log_path=log_path)

        self.best_accuracy = 0
        self.data = {"train": dict(), "valid": dict()}
        # self.data = {"train": dict(), "valid": dict(), "test": dict()}

        self.model = Model()

        self.batch_size = 1
        self.learning_rate = 0.1
        self.regularizer_val = 0.00001

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

    def preprocess(self, text_preprocessor, load_images=False):

        # args example : (BertPreprocessor(pretrained_model_type='bert-base-uncased', do_lower_case=True,
        #                                                                               load_bert=False))

        for key, value in self.data.items():
            value["type"] = key

            text_preprocessor.execute(value)

            if load_images:
                value["image_data"] = np.load("./image_data/" + key + ".npy")

            else:
                value["image_data"] = np.array([cv2.resize(cv2.imread(filename=self.IMAGE_COMPLETE_PATH + img_path),
                                                           self.image_res[0:2]).astype('float32')
                                                for img_path in tqdm(value["img"])])

                np.save("./image_data/" + key + ".npy", value["image_data"])

            value["image_data"] = resnet50.preprocess_input(value["image_data"])

            value["model_input"] = [value["bert_output"][:value["image_data"].shape[0]], value["image_data"]]
            # value["model_input"] = value["bert_output"]
            # value["model_input"] = value["image_data"]

    def __build_image_component(self):

        input_ = Input(shape=self.image_res)

        # x = Lambda(lambda image: tf.image.resize(image, self.image_res[0:2]))(input_)

        resnet_model = ResNet50(include_top=False, weights="imagenet", pooling="max")
        for layer in resnet_model.layers:
            layer.trainable = False

        x = resnet_model(input_)

        x = Flatten()(x)

        output_ = Reshape((1, 768))(Dense(768, activation='relu', kernel_initializer="glorot_normal"
                                          , kernel_regularizer=l2(self.regularizer_val))(x))

        # output_ = Dense(768, activation='relu', kernel_initializer="glorot_normal"
        #                 , kernel_regularizer=l2(self.regularizer_val))(x)

        return input_, output_

    def build(self, *args):

        image_input, image_output = self.__build_image_component()

        text_input = Input(shape=(1, 768), dtype="float32")

        # x = Dense(units=768, kernel_initializer="glorot_normal")(text_input)
        # x = Activation("relu")(x)
        # x = Dropout(0.2)(x)

        x = Add()([text_input, image_output])
        x = BatchNormalization()(x)

        # x = Concatenate()([text_input, image_output])
        #
        # x = image_output

        x = Dense(units=512, kernel_initializer="glorot_normal", kernel_regularizer=l2(self.regularizer_val))(x)
        x = Activation("relu")(x)
        # x = Dropout(0.3)(x)

        x = Dense(units=256, kernel_initializer="glorot_normal", kernel_regularizer=l2(self.regularizer_val))(x)
        x = Activation("relu")(x)
        # x = Dropout(0.3)(x)

        x = Dense(units=128, kernel_initializer="glorot_normal", kernel_regularizer=l2(self.regularizer_val))(x)
        x = Activation("relu")(x)
        # x = Dropout(0.5)(x)

        last_layer = Dense(units=2, activation="relu")(x)

        self.model = Model([text_input, image_input], last_layer)
        # self.model = Model(text_input, last_layer)
        # self.model = Model(image_input, last_layer)

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
