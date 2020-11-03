from Implementation.meme.image import Image
from Implementation.meme.meme_builder import MemeBuilder
from Implementation.meme.textoptions.textcoloroption import TextColorOption
from Implementation.meme.textoptions.textpositionoption import TextPositionOption
from Implementation.meme.textoptions.textsizeoption import TextSizeOption


simple_meme = MemeBuilder() \
    .with_image(Image(r"E:\Facultate\Master\Semestrul 1\ASET\Proiect\data\img\01967.png")) \
    .with_text("New meme text on a simple image, testing the wrapper") \
    .with_option(TextColorOption((100, 10, 256), (0, 0, 0))) \
    .with_option(TextSizeOption(2)) \
    .with_option(TextPositionOption((0, 100))) \
    .build()

simple_meme.show_img()
