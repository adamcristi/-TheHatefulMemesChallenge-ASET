# import time
# from datetime import datetime
#
# from Implementation.classifiers.custom_classifier import CustomClassifier
# from Implementation.preprocess.images.img2vecencoding import Img2VecEncoding
# from Implementation.preprocess.texts.bert_preprocessor_torch import BertPreprocessor
#
# # You need to create a txt file in which you add your path to the json file that contains the text of the memes
# # (that "train.jsonl") and then add the .txt file to .gitignore

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

import time
from datetime import datetime
import platform

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
#
#
# # You also need to have the images in the path ".../Implementation/img/"
# # (but don't put them on github -> add them to .gitignore),
# # and a directory ".../Implementation/encoded_images/" (empty).
#
#
# # Path to the encoding of the images (this example is just for the first 10 memes in the "train.jsonl" file)
# ENCODED_IMG_PATH = "./encoded_images/encoded_full"
# ENCODED_IMG_EXT = ".pickle"
#
# classifier = CustomClassifier(log_path=r".\logging\results")
#
# classifier.load_data((TRAIN_PATH, "train"))
# classifier.load_data((VALID_PATH, "valid"))
# # Test set doesn't have labels for some reason ...
# # classifier.load_data((TEST_PATH, "test"))
#
#
# # "do_lower_case" should be True for uncased-bert and False for cased-bert.
# # "precalculated" means that the images were already encoded and saved in ENCODED_IMG_PATH.
# # If it is False, then it will recalculate the encodings and save them to ENCODED_IMG_PATH
# classifier.preprocess((BertPreprocessor(pretrained_model_type='bert-base-uncased', do_lower_case=True),
#                        Img2VecEncoding(precalculated=True, enc_path=ENCODED_IMG_PATH, ext=ENCODED_IMG_EXT)))
#
# # The .pth extension is specific to pytorch data serialization
# # classifier.load_model("./saved_models/test_model.pth")
# classifier.train((32, 20000, 0.001, "./saved_models/" + str(time.time()) + "_" +
#                   datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + "_"))
# # classifier.test_model()


from Implementation.classifiers.keras_classifier import KerasCustomClassifier
from Implementation.preprocess.texts.bert_preprocessor_keras import BertPreprocessor
from Implementation.preprocess.images.images_preprocessor import ImagePreprocessor

model = KerasCustomClassifier(log_path="./logging/results/")

model.load_data((TRAIN_PATH, "train"))
model.load_data((VALID_PATH, "valid"))

# load_bert - if True, loads the precalculated (and saved) bert outputs
#           - if False, recalculates and saves (LONG TIME ~ 1 hour)
# load_images - loads images from a file with a easy loadable format
#               (if set to false it will load them from raw images and then save - takes more time)
# save_newly_computed_images - if set to True (default), saves the current preprocessed images in a new file
#                            - if set to False, only preprocesses the images
# resize_images_wanted - if set to True, the images are resized to a specified or most common dimension
#                      - if set to False (default), the resize is not applied
# remove_text_from_images_wanted - if set to True, the text is removed from the images
#                                - if set to False (default), the removal is not applied
# dimensions_resized_images - if set (width, height) and remove_text_from_images_wanted is True, the images are resized
#                             to that dimensions
#                           - if None and remove_text_from_images_wanted is True, the images are resized to the most
#                             common dimensions
#                           - if set (width, height) and remove_text_from_images_wanted is False, the resize is not
#                             applied
#                           - if None and remove_text_from_images_wanted is False, the resize is not applied

# model.preprocess(BertPreprocessor(pretrained_model_type='bert-base-uncased', do_lower_case=True, load_bert=True),
#                  load_images=True)

model.preprocess(BertPreprocessor(pretrained_model_type='bert-base-uncased', do_lower_case=True, load_bert=True),
                 ImagePreprocessor(resize_images_wanted=True, dimensions_resized_images=(256, 256),
                                   remove_text_from_images_wanted=True))

epochs = 50
batch_size = 16
learning_rate = 0.001

model.build()
model.train(batch_size, epochs, learning_rate,
            "./saved_models/" + str(time.time()) + "_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") +
            "ResNet50_epochs" + str(epochs) + "_batch" + str(batch_size) + "_lr" + str(learning_rate))

# model.load_model("./saved_models/1608467048.0221534_12_20_2020_14_24_08.h5")
# model.evaluate()

# DEPRECATED ==================================================