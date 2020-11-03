from Implementation.meme.textoptions.textoption import TextOption


class TextPositionOption(TextOption):

    def __init__(self, position):
        super().__init__()
        self.position = position

    def __str__(self):
        return "TextPositionOption object."

    def apply(self, text):
        text.set_pos(self.position)
