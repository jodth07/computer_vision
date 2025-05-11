from typing import Optional

import numpy as np
import requests
import cv2 as cv

BASE_IMAGE_URL = "https://raw.githubusercontent.com/jodth07/computer_vision/develop/resources/"

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

    def show_image(
        self, window_name: str = "Image", input_image: Optional[np.ndarray] = None
    ) -> "ImageProcessor":
        """
        Display the loaded image in a window.
        """
        print(f"Previewing Image in {window_name}")

        image = input_image if input_image is not None else self.image
        if image is None:
            raise ValueError("No image loaded.")
        print(f"Press `q` to continue")

        cv.imshow(window_name, image)
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


if __name__ == '__main__':
    image_file_name = "Mod4CT1.jpg"
    image_path = f"{BASE_IMAGE_URL}{image_file_name}"
    image_processor = ImageProcessor()
    image_processor.load_image_from_url(image_path)
    image_processor.save_image_to_file(image_file_name)
    # image_processor.show_image()

    image = image_processor.image
    print(f"image shape: {image.shape}")

    # Add labels above each filtered image
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (3, 3, 3)  # White text
    thickness = 1

    height, width, channels = image.shape
    canvas = np.ones((height * 3 + 50, width * 4 + 70, channels), dtype=np.uint8) * 255
    height, width, channels = image.shape

    base_h, base_w = height, width
    new_h, new_w = base_h + height, base_w + width
    # canvas[:height, :width] = image

    median_blur = cv.medianBlur(canvas, 3)
    canvas[:height, :width] = median_blur


    image_processor.show_image(window_name="Filtered Image", input_image=canvas)
