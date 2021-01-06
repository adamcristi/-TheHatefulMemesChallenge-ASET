import time
from datetime import datetime
import os


class Logger:

    def __init__(self, log_path="."):
        self.log_path = log_path
        self.model_timestamp = self.get_timestamp()

    @staticmethod
    def get_timestamp():
        return str(int(time.time())) + "_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    def log(self, dict_):
        path = os.path.join(self.log_path, self.model_timestamp)

        with open(path, "a") as file:
            file.write(str(dict_) + "\n")
