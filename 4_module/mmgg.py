import cv2 as cv
import os
import numpy as np
import requests
from requests import Response
from typing import Optional

BASE_IMAGE_URL = "https://raw.githubusercontent.com/jodth07/computer_vision/develop/resources/"

class ImageProcessor:

    def __init__(self):
        self.image = None
        self.resources = set()
        self.shape = None
        self.height = None
        self.width = None
        self.channels = None

    def load_image_from_url(self, url: str):
        """
        Import image from url
        """

        print(f"Downloading Image from {url} to memory")
        response = requests.get(url)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        self.image = image
        self.shape = image.shape
        self.height, self.width, self.channels = image.shape
        return self

    def download_resource(self, ufl) -> str:
        """
        Download the resources from the GitHub repository.
        """
        filename = os.path.basename(ufl)
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            response: Response = requests.get(ufl)
            with open(filename, "wb") as file:
                file.write(response.content)
            print(f"Downloaded {filename}.")
        else:
            print(f"{filename} already exists.")
        self.resources.add(filename)
        return filename

    def load_image_from_file(self, file_path: str) -> "ImageProcessor":
        """
        Load an image from a file path.
        """
        print(f"Loading Image from {file_path} to memory")

        image = cv.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image from {file_path}")
        print(f"Image loaded from {file_path}")
        self.image = image
        self.shape = image.shape
        self.height, self.width, self.channels = image.shape
        return self

    def resize_image(self, value: float) -> "ImageProcessor":
        """
        Resize the image to half or double its size.
        """
        if self.image is None:
            raise ValueError("No image loaded.")

        new_size = (int(self.width * value), int(self.height * value))

        resized_image = cv.resize(self.image, new_size)
        self.image = resized_image
        print(f"Resized image to {new_size}")
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


def apply_filters(image, kernel_size: tuple[int, int]):
    """
    Apply mean, median, and Gaussian filters to the image.
    for Gaussian, select two different values of sigma.
    """
    mean_filter = cv.blur(image, kernel_size)
    median_filter = cv.medianBlur(image, kernel_size[0])
    gaussian_filter_1 = cv.GaussianBlur(image, ksize=kernel_size, sigmaX=0.2)
    gaussian_filter_2 = cv.GaussianBlur(image, ksize=kernel_size, sigmaX=2.0)
    return {
        "mean_filter": mean_filter,
        "median_filter": median_filter,
        "gaussian_filter_1": gaussian_filter_1,
        "gaussian_filter_2": gaussian_filter_2,
    }


def get_height(height, pos):
    n_height = int(height / 4)
    base = (height * (pos - 1)) + n_height
    end = height * pos + n_height
    return base, end


def get_width(width, pos):
    n_width = int(width / 4)
    base = width * pos + n_width
    end = width * (pos + 1) + n_width
    return base, end


def process_and_draw_filters(
    canvas,
    image,
    height,
    width,
    kernel_size,
    row_pos,
    font,
    font_scale,
    color,
    thickness,
):
    """
    Apply filters, draw them on the canvas, and add labels for a specific kernel size and row position.
    """
    base_h, end_h = get_height(height, row_pos)
    filtered_images = apply_filters(image, kernel_size)
    cv.putText(
        canvas, f"{kernel_size}", (10, base_h + 80), font, font_scale, color, thickness
    )
    for idx, (name, filtered_image) in enumerate(filtered_images.items()):
        base_w, end_w = get_width(width, idx)
        canvas[base_h:end_h, base_w:end_w] = filtered_image


def main():
    image_file_name = "Mod4CT1.jpg"
    image_path = f"{BASE_IMAGE_URL}{image_file_name}"
    image_processor = ImageProcessor()
    image_processor.load_image_from_url(image_path)
    image_processor.save_image_to_file(image_file_name)

    image = image_processor.image
    print(f"image shape: {image.shape}")

    # Add labels above each filtered image
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (3, 3, 3)  # White text
    thickness = 1

    height, width, channels = image.shape
    canvas = np.ones((height * 3 + 50, width * 4 + 70, channels), dtype=np.uint8) * 255

    base_h, end_h = get_height(height, 1)
    kernel_size = (3, 3)
    filtered_images = apply_filters(image, kernel_size)
    for idx, (name, filtered_image) in enumerate(filtered_images.items()):
        base_w, end_w = get_width(width, idx)
        cv.putText(
            canvas, name, (base_w + 10, base_h - 5), font, font_scale, color, thickness
        )

    for idx, kernel_size in enumerate([(3, 3), (5, 5), (7, 7)]):
        process_and_draw_filters(
            canvas,
            image,
            height,
            width,
            kernel_size,
            idx + 1,
            font,
            font_scale,
            color,
            thickness,
        )

    image_processor.show_image(window_name="Filtered Image", input_image=canvas)


if __name__ == "__main__":
    main()
