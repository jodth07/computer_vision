import os
from os import environ

import cv2 as cv
import numpy as np
import requests
from matplotlib import pyplot as plt
from dotenv import load_dotenv
# load_dotenv()
IS_DEV = environ.get("IS_DEV", "False").lower() == "true"
BASE_IMAGE_URL = "https://raw.githubusercontent.com/jodth07/computer_vision/develop"

class ImageProcessor:

    def __init__(self):
        self.image = None

    def load_image_from_url(self, url: str):
        """
        Import image from url
        """

        print(f"Downloading Image from {url} to memory")
        response = requests.get(url)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        self.image = image
        return self

    def load_image_from_file(self, file_path: str):
        """
        Load an image from a file path.
        """
        print(f"Loading Image from {file_path} to memory")

        self.image = cv.imread(file_path)
        if self.image is None:
            raise ValueError(f"Could not read image from {file_path}")
        print(f"Image loaded from {file_path}")
        return self

    def show_image(self, window_name: str = "Image"):
        """
        Display the loaded image in a window.
        """
        print(f"Previewing Image in {window_name}")

        if self.image is None:
            raise ValueError("No image loaded.")
        print(f"Press `q` to continue")

        cv.imshow(window_name, self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return self

    def save_image_to_file(self, file_path: str):
        """
        Write a copy of the image to any directory
        """
        if self.image is None:
            raise ValueError("No image loaded.")
        cv.imwrite(file_path, self.image)
        print(f"Image saved to {file_path}")
        return self


"""
Implement an adaptive thresholding scheme to segment the images as best as you can.
"""
def segmenting_thresholding(image):

    # # Apply Gaussian adaptive thresholding
    thresholded_image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv.THRESH_BINARY, 115, 1)
    # thresholded_image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                          cv.THRESH_BINARY, 15, 5)
    return thresholded_image




if __name__ == '__main__':

    print(IS_DEV)

    image_processor = ImageProcessor()

    images = ["6_indoor_scene.png", "6_outdoor_scene.png", "6_closeup_mg.png"]

    # download images
    if not IS_DEV:
        os.makedirs("../resources", exist_ok=True)
        base_image_path = f"{BASE_IMAGE_URL}/resources"
        for image_file_name in images:
            image_processor.load_image_from_url(f"{base_image_path}/{image_file_name}")
            image_processor.save_image_to_file(image_file_name)

    base_image_path = "../resources/"

    """Find on the internet (or use a camera to take) three different types of images:
    an indoor scene, outdoor scenery, and a close-up scene of a single object."""
    indoor_img = cv.imread(f"{base_image_path}/6_indoor_scene.png", 0)
    outdoor_img = cv.imread(f"{base_image_path}/6_outdoor_scene.png", 0)
    closeup_img = cv.imread(f"{base_image_path}/6_closeup_mg.png", 0)


    threshold_indoor = segmenting_thresholding(indoor_img)
    threshold_outdoor = segmenting_thresholding(outdoor_img)
    threshold_close = segmenting_thresholding(closeup_img)

    # Display the results
    titles = ['Indoor', 'Outdoor', 'Close Up']
    images = [threshold_indoor, threshold_outdoor, threshold_close]

    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()