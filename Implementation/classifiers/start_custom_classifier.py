from Implementation.classifiers.custom_classifier import *
import Implementation.monitors.monitor_save_model_automatically
from Implementation.monitors.monitor_save_model_automatically import classifier

if __name__ == "__main__":
    classifier.load_data(args="./test_data.jsonl")
    classifier.preprocess((BertPreprocessor(pretrained_model_type='bert-base-uncased', do_lower_case=True),
                           Img2VecEncoding(enc_path="", precalculated=False)))

    classifier.train(args=(500, 1, 0.1, ''))
    classifier.test_model()
