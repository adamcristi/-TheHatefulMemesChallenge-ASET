from Implementation.meme.textoptions.textoption import TextOption


class TextColorOption(TextOption):

    def __init__(self, inside_color, outside_color):
        super().__init__()
        self.inside_color = inside_color
        self.outside_color = outside_color

    def __str__(self):
        return "TextColorOption object."

    def apply(self, text):
        text.set_inside_color(self.inside_color)
        text.set_outside_color(self.outside_color)



