import torch

from Implementation.preprocess.preprocessor import Preprocessor
from img2vec_pytorch import Img2Vec
from PIL import Image


class Img2VecEncoding(Preprocessor):

    def __init__(self):
        super().__init__()

    def execute(self, data):

        img2vec = Img2Vec(cuda=True)

        data["img_encoded"] = []

        for img_path in data["img"]:

            img = Image.open(img_path).convert('RGB')

            encoded = img2vec.get_vec(img, tensor=True)
            encoded = torch.flatten(encoded)
            encoded = torch.reshape(encoded, (1, -1))

            data["img_encoded"] += [encoded]
