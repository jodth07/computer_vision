import sys
from typing import Optional

import cv2 as cv
import requests
import os
import numpy as np

DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/jodth07/computer_vision/develop/resources/shutterstock130285649--250.jpg"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# image_url = "https://csuglobal.instructure.com/courses/109078/files/8037617?wrap=1"

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

def main(input_file_path: Optional[str] = None):
    image_processor = ImageProcessor()
    """
    import the following from ufl.
    display the image.
    rite a copy of the image to any directory
    """
    url_path = input_file_path or DEFAULT_IMAGE_URL
    loaded_image = image_processor.load_image_from_url(url_path)

    file_name = url_path.split("/")[-1]
    path_split = file_name.split(".")
    new_image_write = ".".join(path_split[:-1]) + "_copy." + path_split[-1]

    loaded_image.show_image()
    loaded_image.save_image_to_file(new_image_write)


def get_arguments() -> Optional[str]:
    args = sys.argv[1:]
    if len(args) > 0:
        return args[0]
    else:
        print("No Argument passed in, downloading default image from URL")
        return None

if __name__ == "__main__":
    file_url = get_arguments()
    main(file_url)
