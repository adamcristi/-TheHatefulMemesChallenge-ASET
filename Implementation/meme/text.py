import textwrap
from cv2 import *
import numpy as np


class Text:

    def __init__(self, text="", img_sizes=(0, 0)):
        img_width, img_height = img_sizes
        print(img_sizes)
        self.text = text

        self.font = 0
        self.font_scale = 1
        self.font_thickness = 2

        self.__get_text_sizes()
        self.text_width, self.text_height = self.__get_text_sizes()

        self.character_size_aprox = np.max([self.text_width // len(self.text) - 1, 1])

        self.margin_space = 20

        self.max_line_characters = (img_width - 2 * self.margin_space) // self.character_size_aprox

        self.wrapped_text = textwrap.wrap(self.text, width=self.max_line_characters)

        self.space_between_lines = 10

        self.margin_space_bottom = 10

        self.x_pos = 10

        self.y_pos = img_height - \
                     len(self.wrapped_text) * (self.text_height + self.space_between_lines + 2) - \
                     self.margin_space_bottom

    def __str__(self):
        return self.text

    def __get_text_sizes(self):
        return cv2.getTextSize(self.text, self.font, self.font_scale, self.font_thickness + 3)[0]

    def compute_text_sizes(self):
        self.text_width, self.text_height = self.__get_text_sizes()

    def get_pos(self):
        return self.x_pos, self.y_pos

    def get_font_data(self):
        return self.font, self.font_scale, self.font_thickness

    def get_text_sizes(self):
        return self.text_width, self.text_height

    def get_margin_space(self):
        return self.margin_space

    def get_margin_space_bottom(self):
        return self.margin_space_bottom
