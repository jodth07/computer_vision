import sys
from typing import Optional

import cv2 as cv
import requests
import os
import numpy as np

DEFAULT_IMAGE_URL = "https://csuglobal.instructure.com/courses/109078/files/8037594/download?download_frd=1"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# image_url = "https://csuglobal.instructure.com/courses/109078/files/8037617?wrap=1"

class ImageProcessor:

    def __init__(self):
        self.image = None
        self.blue = None
        self.green = None
        self.red = None
        # self.shape = None

    def load_image_from_url(self, url: str):
        """
        Import image from url
        """

        print(f"Downloading Image from {url} to memory")
        response = requests.get(url)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        # self.shape = image.shape
        self.image = image
        return self

    def load_image_from_file(self, file_path: str):
        """
        Load an image from a file path.
        """
        print(f"Loading Image from {file_path} to memory")

        image = cv.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image from {file_path}")
        print(f"Image loaded from {file_path}")
        self.image = image
        # self.shape = image.shape
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
        # if self.image is None:
        #     raise ValueError("No image loaded.")
        cv.imwrite(file_path, self.image)
        print(f"Image saved to {file_path}")
        return self

    def extract_channels(self):
        """
        Extract the RGB channels from the image.
        """
        if self.image is None:
            raise ValueError("No image loaded.")
        # Split the image into its RGB channels
        self.blue, self.green, self.red = cv.split(self.image)
        return self.blue, self.green, self.red

    def merge_channels(self, channels: Optional[tuple] = None):
        """
        merge all these images back into a colored 3D image
        """
        if channels is None:
            if self.blue is None or self.green is None or self.red is None:
                raise ValueError("No channels to merge.")
            # Merge the channels back into a single image
            return cv.merge((self.blue, self.green, self.red))
        else:
            # Merge the channels back into a single image
            if len(channels) > 2:
                return cv.merge(channels)
            else:
                raise ValueError("Not enough channels to merge.")


def main(input_file_path: Optional[str] = None):
    image_processor = ImageProcessor()
    # loaded_image = image_processor.load_image_from_url(DEFAULT_IMAGE_URL)
    input_file_path = "shutterstock-dog--250.jpg"
    loaded_image = image_processor.load_image_from_file(input_file_path)
    # new_image_write = os.path.join(CURRENT_DIR, "shutterstock93075775--250_copy.jpg")
    b, g, r = loaded_image.extract_channels()
    new_merged_image = loaded_image.merge_channels(channels=(b, g, r))
    new_merged_image2 = loaded_image.merge_channels(channels=(g, r, b))
    # cv.imshow("Image", image_split)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cv.imshow('Channels', new_merged_image2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    print(new_merged_image2.shape)

    # loaded_image.show_image()
    # loaded_image.save_image_to_file(new_image_write)


def get_arguments() -> Optional[str]:
    args = sys.argv[1:]
    if len(args) > 0:
        return args[0]
    else:
        print("No Argument passed in, downloading default image from URL")
        return None

if __name__ == "__main__":
    # input_file_path = get_arguments()
    main()
