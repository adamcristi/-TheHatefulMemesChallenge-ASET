import textwrap
from cv2 import *
import numpy as np


class Text:

    def __init__(self, text="", img_sizes=(0, 0)):
        self.__img_width, self.__img_height = img_sizes

        self.text = text

        self.inside_color = (255, 255, 255)
        self.outside_color = (0, 0, 0)

        self.font = 0
        self.font_scale = 1
        self.font_thickness = 2

        self.text_width, self.text_height = self.__get_text_sizes()

        self.character_size_aprox = np.max([self.text_width // len(self.text) - 1, 1])

        self.margin_space = 20

        self.max_line_characters = (self.__img_width - 2 * self.margin_space) // self.character_size_aprox

        self.wrapped_text = textwrap.wrap(self.text, width=self.max_line_characters)

        self.space_between_lines = 10

        self.margin_space_bottom = 10

        self.x_pos = 10

        self.y_pos = self.__img_height - \
                     len(self.wrapped_text) * (self.text_height + self.space_between_lines + 2) - \
                     self.margin_space_bottom

    def __str__(self):
        return self.text

    def __get_text_sizes(self):
        return cv2.getTextSize(self.text, self.font, self.font_scale, self.font_thickness + 3)[0]

    def compute_text_sizes(self):
        self.text_width, self.text_height = self.__get_text_sizes()

        self.character_size_aprox = np.max([self.text_width // len(self.text) - 1, 1])

        self.max_line_characters = (self.__img_width - 2 * self.margin_space) // self.character_size_aprox

        self.wrapped_text = textwrap.wrap(self.text, width=self.max_line_characters)

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

    def get_inside_color(self):
        return self.inside_color

    def get_outside_color(self):
        return self.outside_color

    def set_pos(self, pos):
        self.x_pos, self.y_pos = pos

    def set_font_data(self, font_data):
        font, font_scale, font_thickness = font_data
        if font > -1:
            self.font = font
        if font_scale > -1:
            self.font_scale = font_scale
        if font_thickness > -1:
            self.font_thickness = font_thickness

    def set_inside_color(self, inside_color):
        self.inside_color = inside_color

    def set_outside_color(self, outside_color):
        self.outside_color = outside_color
