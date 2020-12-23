import cv2
import os
import numpy as np

from Implementation.preprocess.preprocessor import Preprocessor
from collections import Counter
from tqdm import tqdm

IMAGES_COMPLETE_PATH = "./data/data/"
SAVED_PREPROCESSED_IMAGES_COMPLETE_PATH = "./image_data/"
SAVED_PREPROCESSED_IMAGES_EXTENSION = ".npy"
MINIMUM_CONTOUR_AREA = 35.0
MINIMUM_WHITE_PIXELS_MASK_CONTOUR = 15


class ImagePreprocessor(Preprocessor):

    def __init__(self, load_images=False, save_newly_computed_images=True,
                 resize_images_wanted=False, remove_text_from_images_wanted=False, dimensions_resized_images=None):
        super().__init__()

        self.load_images = load_images
        self.save_newly_computed_images = save_newly_computed_images
        self.have_to_resize_images = resize_images_wanted
        self.have_to_remove_text_from_images = remove_text_from_images_wanted
        self.specified_resize_dimensions = dimensions_resized_images

        if self.have_to_resize_images and self.specified_resize_dimensions is None:
            self.images_frequently_height = None
            self.images_frequently_width = None

            self.get_frequently_dimensions_of_images()

    def get_frequently_dimensions_of_images(self):
        all_images_height = []
        all_images_width = []

        for root, dirs, files in os.walk(IMAGES_COMPLETE_PATH + "img"):
            for file in tqdm(files):
                path_image = os.path.join(root, file)
                image = cv2.imread(path_image)
                all_images_height.append(image.shape[0])
                all_images_width.append(image.shape[1])

        self.images_frequently_height = Counter(all_images_height).most_common(1)[0][0]
        self.images_frequently_width = Counter(all_images_width).most_common(1)[0][0]

    @staticmethod
    def resize_image(original_image, new_width, new_height):
        current_width, current_height = original_image.shape[:2]
        # same
        if current_width == new_width and current_height == new_height:
            resized_image = original_image
        # shrink image
        elif current_width > new_width and current_height > new_height:
            resized_image = cv2.resize(original_image, (int(new_width), int(new_height)), interpolation=cv2.INTER_AREA)
        # completely or partially enlarge image
        else:
            resized_image = cv2.resize(original_image, (int(new_width), int(new_height)), interpolation=cv2.INTER_LINEAR)
        return resized_image

    @staticmethod
    def remove_text_from_image(original_image):

        # convert image to grayscale
        grayscale_transformed_image = cv2.cvtColor(original_image, code=cv2.COLOR_BGR2GRAY)

        # apply bilateral filter for smoothing the image
        blur_transformed_image = cv2.bilateralFilter(grayscale_transformed_image, d=5, sigmaColor=75, sigmaSpace=75)

        # calculate morphological gradient
        kernel_gradient = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))
        gradient_transformed_image = cv2.morphologyEx(blur_transformed_image, op=cv2.MORPH_GRADIENT,
                                                      kernel=kernel_gradient)

        # apply threshold (binarization)
        threshold_used_in_process, threshold_transformed_image = cv2.threshold(gradient_transformed_image, thresh=0.0,
                                                                               maxval=255.0,
                                                                               type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # calculate morphological closing
        kernel_closing = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 1))
        closing_transformed_image = cv2.morphologyEx(threshold_transformed_image, op=cv2.MORPH_CLOSE,
                                                     kernel=kernel_closing)

        # find the contours
        contours_transformed_image, contours_hierarchy = cv2.findContours(closing_transformed_image,
                                                                          mode=cv2.RETR_EXTERNAL,
                                                                          method=cv2.CHAIN_APPROX_NONE)

        # get mask of text from image
        mask_transformed_image = np.zeros(closing_transformed_image.shape, dtype=np.uint8)
        mask_white_pixels_text_image = np.zeros(closing_transformed_image.shape, dtype=np.uint8)

        white_pixels_original_image = np.logical_and(np.logical_and((original_image[:, :, 0] == 255),
                                                                    (original_image[:, :, 1] == 255)),
                                                     np.logical_and((original_image[:, :, 1] == 255),
                                                                    (original_image[:, :, 2] == 255)))

        for index, contour in enumerate(contours_transformed_image):
            area_current_contour = cv2.contourArea(contour)
            if area_current_contour > MINIMUM_CONTOUR_AREA:
                x, y, width, height = cv2.boundingRect(contour)
                cv2.drawContours(mask_transformed_image, contours_transformed_image, index, (255, 255, 255), -1)

                mask_current_contour = np.logical_and(white_pixels_original_image[y:y + height, x:x + width],
                                                      (mask_transformed_image[y:y + height, x:x + width] == 255))
                mask_current_contour = np.where(mask_current_contour == True, 255, 0)

                if np.count_nonzero(mask_current_contour == 255) >= MINIMUM_WHITE_PIXELS_MASK_CONTOUR:
                    mask_white_pixels_text_image[y:y + height, x:x + width] = np.copy(mask_current_contour)

        # calculate morphological dilation
        kernel_dilatation = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(20, 20))
        dilatation_mask_white_pixels_text_image = cv2.morphologyEx(mask_white_pixels_text_image, op=cv2.MORPH_DILATE,
                                                                   kernel=kernel_dilatation)

        # inpaint original image
        new_image_without_text = cv2.inpaint(original_image, inpaintMask=dilatation_mask_white_pixels_text_image,
                                             inpaintRadius=15, flags=cv2.INPAINT_NS)

        return new_image_without_text

    def get_filename_of_preprocessed_images(self, data_key):
        filename = "images_preprocessed_{}".format(data_key)
        if self.have_to_resize_images:
            filename += "_resized"
            if self.specified_resize_dimensions is None:
                filename += "_{}_{}".format(str(self.images_frequently_width), str(self.images_frequently_height))
            else:
                filename += "_{}_{}".format(str(self.specified_resize_dimensions[0]),
                                            str(self.specified_resize_dimensions[1]))
        if self.have_to_remove_text_from_images:
            filename += "_without_text"

        return filename

    def execute(self, data, data_key):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')

        if self.load_images:
            data["image_data"] = np.load(
                SAVED_PREPROCESSED_IMAGES_COMPLETE_PATH + self.get_filename_of_preprocessed_images(data_key) +
                SAVED_PREPROCESSED_IMAGES_EXTENSION)

        else:
            data["image_data"] = []

            for image_path in tqdm(data["img"]):
                current_image = cv2.imread(IMAGES_COMPLETE_PATH + image_path)

                if self.have_to_remove_text_from_images:
                    current_image = self.remove_text_from_image(original_image=current_image)

                if self.have_to_resize_images:
                    if self.specified_resize_dimensions is None:
                        current_image = self.resize_image(original_image=current_image,
                                                          new_width=self.images_frequently_width,
                                                          new_height=self.images_frequently_height)
                    else:
                        current_image = self.resize_image(original_image=current_image,
                                                          new_width=self.specified_resize_dimensions[0],
                                                          new_height=self.specified_resize_dimensions[1])

                data["image_data"] += [current_image]

            data["image_data"] = np.array([data["image_data"]], dtype="float32")

            if self.save_newly_computed_images:
                np.save(SAVED_PREPROCESSED_IMAGES_COMPLETE_PATH + self.get_filename_of_preprocessed_images(data_key) +
                        SAVED_PREPROCESSED_IMAGES_EXTENSION, data["image_data"])
