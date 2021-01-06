from Implementation.classifiers.pytorch_classifier import *
import Implementation.monitors.monitor_save_model_automatically_pytorch_classifier
from Implementation.monitors.monitor_save_model_automatically_pytorch_classifier import classifier_pytorch
from pathlib import Path
import os

ROOT_FILENAME = "Implementation"
ROOT_DIRECTORY = Path(__file__)
while str(ROOT_DIRECTORY.name) != ROOT_FILENAME:
    ROOT_DIRECTORY = ROOT_DIRECTORY.parent

if __name__ == "__main__":
    # classifier_pytorch.load_data(args=(os.path.join(ROOT_DIRECTORY, 'data', 'data', 'train_reduced.jsonl'), "train"))
    # classifier_pytorch.load_data(args=(os.path.join(ROOT_DIRECTORY, 'data', 'data', 'dev_seen_reduced.jsonl'), "valid"))

    classifier_pytorch.load_data(
        args=(os.path.join(ROOT_DIRECTORY, 'data', 'data', 'monitors_testing_train_data.jsonl'), "train"))
    classifier_pytorch.load_data(
        args=(os.path.join(ROOT_DIRECTORY, 'data', 'data', 'monitors_testing_valid_data.jsonl'), "valid"))

    classifier_pytorch.preprocess((BertPreprocessor(pretrained_model_type='bert-base-uncased', do_lower_case=True),
                                   Img2VecEncoding(enc_path="", ext="", precalculated=False)))

    classifier_pytorch.train(args=(1, 1, 0.1, ''))
    classifier_pytorch.test_model(data_used="valid")
