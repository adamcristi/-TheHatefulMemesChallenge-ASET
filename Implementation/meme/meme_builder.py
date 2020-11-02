from Implementation.meme.meme import *


class Option:

    def __init__(self):
        pass

    def apply(self):
        pass


class OptionExample(Option):

    def __init__(self, param):
        super().__init__()

        self.param = param

    def __str__(self):

        return str(self.param)


class MemeBuilder:

    def __init__(self, options=None):

        if options is None:
            options = []

        self.meme = Meme()
        self.options = options

    def get_options(self):

        return self.options

    def get_image(self):

        return self.meme.get_image()

    def get_text(self):

        return self.meme.get_text()

    def with_image(self, image):

        self.meme.set_image(image)
        return self

    def with_text(self, text):

        self.meme.set_text(text)
        return self

    def with_option(self, option):

        self.options += [option]
        return self

    def build(self):

        # TODO implement meme creation

        return self.meme

