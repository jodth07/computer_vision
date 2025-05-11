"""
Acquire an gray_scaled of a latent fingerprint. In OpenCV, 
write algorithms to process the gray_scaled using morphological operations 
(dilation, erosion, opening, and closing).
"""

import cv2 as cv
import requests
import numpy as np
from matplotlib import pyplot as plt

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


def morphological_operations(image_path):
    # Read the gray_scaled
    image = cv.imread(image_path, 1)
    gray_scaled = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Thresholding the gray_scaled
    _, thresh = cv.threshold(gray_scaled, 127, 255, cv.THRESH_BINARY)

    # Define a kernel for morphological operations
    k_val = 3
    kernel = np.ones((k_val, k_val), np.uint8)

    # Perform morphological operations
    dilation = cv.dilate(gray_scaled, kernel, iterations=1)
    erosion = cv.erode(gray_scaled, kernel, iterations=1)
    opening = cv.morphologyEx(gray_scaled, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(gray_scaled, cv.MORPH_CLOSE, kernel)

    # Display the results
    titles = ['Original Image', 'Thresholded Image', 'Dilation', 'Erosion', 'Opening', 'Closing']
    images = [gray_scaled, thresh, dilation, erosion, opening, closing]

    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    image_file_name = "fingerprint1.png"
    image_path =  f"https://raw.githubusercontent.com/jodth07/computer_vision/develop/resources/{image_file_name}"
    image_processor = ImageProcessor()
    image_processor.load_image_from_url(image_path)
    image_processor.save_image_to_file(image_file_name)
    image_processor.show_image()
    morphological_operations(image_file_name)

    # morphological_operations(image_path)