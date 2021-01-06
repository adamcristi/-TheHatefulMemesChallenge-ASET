from Implementation.classifiers.keras_classifier import KerasCustomClassifier
from Implementation.preprocess.texts.bert_preprocessor_keras import BertPreprocessor
from Implementation.preprocess.images.images_preprocessor import ImagePreprocessor
from Implementation.monitors.monitor_save_model_automatically_keras_classifier import classifier_keras
from pathlib import Path
import os

ROOT_FILENAME = "Implementation"
ROOT_DIRECTORY = Path(__file__)
while str(ROOT_DIRECTORY.name) != ROOT_FILENAME:
    ROOT_DIRECTORY = ROOT_DIRECTORY.parent


if __name__ == "__main__":
    #classifier_keras.load_data((os.path.join(ROOT_DIRECTORY, 'data', 'data', 'train_reduced.jsonl'), "train"))
    #classifier_keras.load_data((os.path.join(ROOT_DIRECTORY, 'data', 'data', 'dev_seen_reduced.jsonl'), "valid"))

    classifier_keras.load_data((os.path.join(ROOT_DIRECTORY, 'data', 'data', 'monitors_testing_train_data.jsonl'), "train"))
    classifier_keras.load_data((os.path.join(ROOT_DIRECTORY, 'data', 'data', 'monitors_testing_valid_data.jsonl'), "valid"))

    classifier_keras.preprocess(BertPreprocessor(pretrained_model_type='bert-base-uncased', do_lower_case=True, load_bert=True),
                                ImagePreprocessor(resize_images_wanted=True, dimensions_resized_images=(256, 256),
                                       remove_text_from_images_wanted=False, save_newly_computed_images=False))

    epochs = 1
    classifier_keras.train(epochs, "")

    classifier_keras.evaluate()

