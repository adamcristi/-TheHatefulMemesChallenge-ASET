import cv2


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

    def show_img(self):
        cv2.imshow("img", self.image.get_img())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
