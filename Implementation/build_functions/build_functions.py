
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation, Input, GlobalMaxPooling2D, Conv2D, MaxPooling2D, \
    GlobalAveragePooling2D, Flatten, Add, Lambda, Concatenate, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Model

from tensorflow.keras.regularizers import l2

from tensorflow.keras.initializers import he_normal

from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import ResNet50

import numpy as np

from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger


from Implementation.classifiers.classifier import Classifier


def build_image_component_1(image_res):
    input_ = Input(shape=image_res)

    # x = Lambda(lambda image: tf.image.resize(image, image_res[0:2]))(input_)

    resnet_model = ResNet50(include_top=False, weights="imagenet", pooling="max")
    for layer in resnet_model.layers:
        layer.trainable = False

    x = resnet_model(input_)

    x = Flatten()(x)

    # x = Dense(768, kernel_initializer=he_normal(),
    #           kernel_regularizer=l2(regularizer_val))(x)
    # x = Activation("elu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)

    # output_ = Reshape((1, 768))(x)
    output_ = Reshape((1, 2048))(x)
    # output_ = x

    return input_, output_


def build_1(image_res, regularizer_val, learning_rate):

    image_input, image_output = build_image_component_1(image_res)

    text_input = Input(shape=(1, 768), dtype="float32")

    # x = Dense(units=768, kernel_initializer=he_normal())(text_input)
    # x = Activation("elu")(x)
    # x = Dropout(0.2)(x)

    # x = Add()([text_input, image_output])
    # x = BatchNormalization()(x)

    x = Concatenate()([text_input, image_output])

    # x = image_output

    x = Dense(units=512, kernel_initializer=he_normal(), kernel_regularizer=l2(regularizer_val))(x)
    x = Activation("elu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(units=256, kernel_initializer=he_normal(), kernel_regularizer=l2(regularizer_val))(x)
    x = Activation("elu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(units=128, kernel_initializer=he_normal(), kernel_regularizer=l2(regularizer_val))(x)
    x = Activation("elu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    last_layer = Dense(units=2, activation="elu")(x)

    model = Model([text_input, image_input], last_layer)
    # model = Model(text_input, last_layer)
    # model = Model(image_input, last_layer)

    print(model.summary())

    optimizer = Adam(learning_rate=learning_rate)
    loss = BinaryCrossentropy()
    metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
