import cv2 as cv
import os

import numpy as np
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/jodth07/computer_vision/main/resources/shutterstock-dog--250.jpg"


if __name__ == '__main__':
    """
    Import this image_colored into OpenCV and 
    extract each of these channels separately to
    create 2D images. 
    merge all these images back into a colored 3D image_colored.
    exchange the reds with the greens? 
    swapping out the blue channel with the red channel (GRB).
    """
    response = requests.get(DEFAULT_IMAGE_URL)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

    image_colored = cv.imdecode(image_array, cv.IMREAD_COLOR)

    cv.namedWindow('image_colored', cv.WINDOW_AUTOSIZE)
    cv.imshow('image_colored', image_colored)
    print(f"image_colored shape {image_colored.shape}")
    print(f"image_colored size {image_colored.size}")
    cv.waitKey(0)

    # Extract each channel
    blue_channel = image_colored[:, :, 0]
    green_channel = image_colored[:, :, 1]
    red_channel = image_colored[:, :, 2]

    cv.imshow('blue_channel', blue_channel)
    print(f"blue_channel shape {blue_channel.shape}")
    cv.waitKey(0)

    cv.imshow('green_channel', green_channel)
    print(f"green_channel shape {green_channel.shape}")
    cv.waitKey(0)

    cv.imshow('red_channel', red_channel)
    print(f"red_channel shape {red_channel.shape}")
    cv.waitKey(0)

    # Merge all channels back into a colored image
    merged_image = cv.merge((blue_channel, green_channel, red_channel))
    cv.imshow('merged_image', merged_image)
    print(f"merged_image shape {merged_image.shape}")
    cv.waitKey(0)

    # Swap the red and green channels
    swap_red_n_green_image = cv.merge((blue_channel, red_channel, green_channel))
    cv.imshow('swap_red_n_green_image', swap_red_n_green_image)
    print(f"swap_red_n_green_image shape {swap_red_n_green_image.shape}")
    cv.waitKey(0)

    # Swap the blue and red channels
    swap_blue_n_red_image = cv.merge((red_channel, green_channel, blue_channel))
    cv.imshow('swap_blue_n_red_image', swap_blue_n_red_image)
    print(f"swap_blue_n_red_image shape {swap_blue_n_red_image.shape}")
    cv.waitKey(0)

    cv.destroyAllWindows()
