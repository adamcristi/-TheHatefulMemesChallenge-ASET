from Implementation.meme.textoptions.textoption import TextOption


class TextSizeOption(TextOption):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __str__(self):
        return "TextSizeOption object."

    def apply(self, text):
        text.set_font_data((-1, self.size, -1))
