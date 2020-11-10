import cv2

from Implementation.meme.meme import *
from Implementation.meme.text import Text


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
        img = self.meme.get_image()
        if img is not None:
            self.meme.set_text(Text(text, img.get_sizes()))
        else:
            raise Exception("Builder needs to get the image before the text!")
        return self

    def with_option(self, option):
        self.options += [option]
        return self

    def build(self):

        text = self.meme.get_text()
        img = self.meme.get_image().get_img()

        for opt in self.options:
            opt.apply(text)

        text.compute_text_sizes()

        x_pos, y_pos = text.get_pos()
        font, font_scale, font_thickness = text.get_font_data()
        inside_color, outside_color = text.get_inside_color(), text.get_outside_color()

        for index, text_line in enumerate(text.wrapped_text):
            y_pos += text.text_height + text.space_between_lines

            cv2.putText(img, text_line, (x_pos, y_pos), font, font_scale, outside_color, font_thickness + 3, cv2.LINE_AA)
            cv2.putText(img, text_line, (x_pos, y_pos), font, font_scale, inside_color, font_thickness, cv2.LINE_AA)

        return self.meme
