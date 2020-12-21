import tensorflow as tf

import time
from datetime import datetime
import platform

from Implementation.build_functions.build_functions import build_1
from Implementation.classifiers.keras_classifier import KerasCustomClassifier
from Implementation.preprocess.texts.bert_preprocessor_keras import BertPreprocessor
from Implementation.preprocess.images.images_preprocessor import ImagePreprocessor

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

with open("meme_text_path.txt", "r") as path_file:
    # .rstrip() -> without new lines at the end
    if str(platform.system()) == "Windows":
        TRAIN_PATH = path_file.readline().rstrip()
        VALID_PATH = path_file.readline().rstrip()
        TEST_PATH = path_file.readline().rstrip()
    elif str(platform.system()) == "Linux" or str(platform.system()) == "Darwin":
        TRAIN_PATH = path_file.readline().rstrip().replace("\\", "/")
        VALID_PATH = path_file.readline().rstrip().replace("\\", "/")
        TEST_PATH = path_file.readline().rstrip().replace("\\", "/")

# Get preprocessed data
temp_model = KerasCustomClassifier()
temp_model.load_data((TRAIN_PATH, "train"))
temp_model.load_data((VALID_PATH, "valid"))
data_dict = temp_model.preprocess(BertPreprocessor(pretrained_model_type='bert-base-uncased',
                                                   do_lower_case=True,
                                                   load_bert=True),
                                  ImagePreprocessor(resize_images_wanted=True,
                                                    dimensions_resized_images=(256, 256)))


def train_config(data, img_model_name, build_function, epochs, batch_size, learning_rate, regularizer_val):

    model = KerasCustomClassifier(log_path="./logging/results/",
                                  build_function=build_function,
                                  batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  regularizer_val=regularizer_val)

    model.data = data

    model.train(epochs,
                "./saved_models/" + str(time.time()) + "_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") +
                img_model_name + "_epochs" + str(epochs) + "_batch" + str(batch_size) + "_lr" + str(learning_rate) +
                "_regularizerval" + str(regularizer_val))

    return model


train_config(data_dict,
             img_model_name="ResNet50",
             build_function=build_1,
             epochs=50,
             batch_size=32,
             learning_rate=0.0001,
             regularizer_val=0.00001)