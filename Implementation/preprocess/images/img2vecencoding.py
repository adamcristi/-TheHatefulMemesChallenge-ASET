import torch
import pickle

from Implementation.preprocess.preprocessor import Preprocessor
from img2vec_pytorch import Img2Vec
from PIL import Image


class Img2VecEncoding(Preprocessor):

    def __init__(self, enc_path, precalculated):
        super().__init__()

        self.precalculated = precalculated
        self.enc_path = enc_path

    def execute(self, data):

        if self.precalculated:

            with open(self.enc_path, "rb") as file:
                data["img_encoded"] = pickle.load(file)

        else:
            img2vec = Img2Vec(cuda=True)

            data["img_encoded"] = []

            for img_path in data["img"]:
                img = Image.open(img_path).convert('RGB')

                encoded = img2vec.get_vec(img, tensor=True)
                encoded = torch.flatten(encoded)
                encoded = torch.reshape(encoded, (1, -1))

                data["img_encoded"] += [encoded]

            with open(self.enc_path, "wb") as file:
                pickle.dump(data["img_encoded"], file)
