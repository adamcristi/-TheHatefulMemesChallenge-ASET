import torch
import pickle
import os

from tqdm import tqdm
from pathlib import Path
from Implementation.preprocess.preprocessor import Preprocessor
from img2vec_pytorch import Img2Vec
from PIL import Image

ROOT_FILENAME = "Implementation"
IMAGE_COMPLETE_PATH_FROM_ROOT = ['data', 'data']


class Img2VecEncoding(Preprocessor):

    def __init__(self, enc_path, ext, precalculated):
        super().__init__()

        self.root_directory = Path(__file__)
        while str(self.root_directory.name) != ROOT_FILENAME:
            self.root_directory = self.root_directory.parent

        self.precalculated = precalculated
        self.enc_path = enc_path
        self.ext = ext

    def execute(self, data):

        if self.precalculated:

            with open(self.enc_path + "_" + data["type"] + self.ext, "rb") as file:
                data["img_encoded"] = pickle.load(file)

        else:
            if torch.cuda.is_available():
                img2vec = Img2Vec(cuda=True)
            else:
                img2vec = Img2Vec()

            data["img_encoded"] = []

            for img_path in tqdm(data["img"]):
                img = Image.open(os.path.join(self.root_directory, *IMAGE_COMPLETE_PATH_FROM_ROOT, img_path)).convert(
                    'RGB')

                encoded = img2vec.get_vec(img, tensor=True)
                encoded = torch.flatten(encoded)
                encoded = torch.reshape(encoded, (1, -1))

                data["img_encoded"] += [encoded]

            if self.enc_path != "":
                with open(self.enc_path + "_" + data["type"] + self.ext, "wb") as file:
                    pickle.dump(data["img_encoded"], file)
