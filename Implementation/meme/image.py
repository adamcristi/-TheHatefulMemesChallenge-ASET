from cv2 import *


class Image:

    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.height, self.width, self.channel_size = self.img.shape

    def get_img(self):
        return self.img

    def get_sizes(self):
        return self.width, self.height

    def get_channel_size(self):
        return self.channel_size
