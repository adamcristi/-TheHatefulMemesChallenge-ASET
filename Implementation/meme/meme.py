from cv2 import *


class Image:

    def __init__(self):
        pass


class Text:

    def __init__(self, text=""):

        self.text = text

    def __str__(self):

        return self.text


class Meme:

    def __init__(self, image=None, text=None):

        self.image = image
        self.text = text

    def get_image(self):

        return self.image

    def set_image(self, image):

        self.image = image

    def get_text(self):

        return self.text

    def set_text(self, text):

        self.text = text