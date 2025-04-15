import cv2 as cv
import requests
import os
import numpy as np


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


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    new_image_write = os.path.join(current_dir, "shutterstock93075775--250-wt.jpg")

    image_url = "https://raw.githubusercontent.com/jodth07/computer_vision/main/resources/shutterstock130285649--250.jpg"
    # image_url = "https://csuglobal.instructure.com/courses/109078/files/8037617?wrap=1"
    image_processor = ImageProcessor()
    loaded_image = image_processor.load_image_from_url(image_url)
    loaded_image.show_image()
    loaded_image.save_image_to_file(new_image_write)
