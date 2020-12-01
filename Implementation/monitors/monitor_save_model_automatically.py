from Implementation.monitors.pythonrv_python3 import rv
from Implementation.classifiers.custom_classifier import CustomClassifier
from datetime import datetime

classifier = CustomClassifier()


@rv.monitor(train_model=classifier.train, test_model=classifier.test_model, save_model=classifier.save_model)
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
            current_date_and_time = now.strftime("%Y-%m-%d_%H:%M:%S")
            classifier.SAVE_PATH = "../automatically_saved_models/model_from_{}.pth".format(current_date_and_time)
            classifier.save_model()

            print("Model saved automatically!")
