from Implementation.monitors.pythonrv_python3 import rv
from Implementation.classifiers.pytorch_classifier import PytorchCustomClassifier
from datetime import datetime
from pathlib import Path
import os

ROOT_FILENAME = "Implementation"
ROOT_DIRECTORY = Path(__file__)
while str(ROOT_DIRECTORY.name) != ROOT_FILENAME:
    ROOT_DIRECTORY = ROOT_DIRECTORY.parent

classifier_pytorch = PytorchCustomClassifier(log_path=os.path.join(ROOT_DIRECTORY, 'logging'))


@rv.monitor(train_model=classifier_pytorch.train, test_model=classifier_pytorch.test_model, save_model=classifier_pytorch.save_model)
@rv.spec(when=rv.POST, history_size=5)
def spec_save_automatically(event):
    if event.fn.test_model.called:
        train_model_done = False
        save_model_done = False
        test_model_done_after_train = False

        for old_event in event.history:
            if old_event.called_function == old_event.fn.train_model:
                train_model_done = True
            elif train_model_done and old_event.called_function == old_event.fn.save_model:
                save_model_done = True
            elif train_model_done and old_event.called_function == old_event.fn.test_model:
                test_model_done_after_train = True

        if train_model_done and test_model_done_after_train and not save_model_done:
            now = datetime.now()
            current_date_and_time = now.strftime("%Y_%m_%d_%H_%M_%S")

            classifier_pytorch.save_model(path=os.path.join(ROOT_DIRECTORY, 'automatically_saved_models',
                                                          'model_from_{}'.format(current_date_and_time)), ext=".pth")

            print("Model saved automatically!")