# import time
# from datetime import datetime
#
# from Implementation.classifiers.custom_classifier import CustomClassifier
# from Implementation.preprocess.images.img2vecencoding import Img2VecEncoding
# from Implementation.preprocess.texts.bert_preprocessor_torch import BertPreprocessor
#
# # You need to create a txt file in which you add your path to the json file that contains the text of the memes
# # (that "train.jsonl") and then add the .txt file to .gitignore
import time
from datetime import datetime

with open("meme_text_path.txt", "r") as path_file:
    # .rstrip() -> without new lines at the end
    TRAIN_PATH = path_file.readline().rstrip()
    VALID_PATH = path_file.readline().rstrip()
    TEST_PATH = path_file.readline().rstrip()
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

model = KerasCustomClassifier(log_path=r".\logging\results")

model.load_data((TRAIN_PATH, "train"))
model.load_data((VALID_PATH, "valid"))

# load_bert - if True, loads the precalculated (and saved) bert outputs
#           - if False, recalculates and saves (LONG TIME ~ 1 hour)
# load_images - loads images from a file with a easy loadable format
#               (if set to false it will load them from raw images and then save - takes more time)
model.preprocess(BertPreprocessor(pretrained_model_type='bert-base-uncased', do_lower_case=True, load_bert=True),
                 load_images=True)

# model.build()
model.train(128, 20, 0.001, "./saved_models/" + str(time.time()) + "_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
