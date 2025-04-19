import sys
from typing import Optional

import cv2 as cv
import requests
import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# image = cv.imread('shutterstock-cat--250.jpg')
#
# cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
# cv.imshow('image', image)
# cv.waitKey(0)
# cv.destroyAllWindows()


if __name__ == '__main__':
    image_path = os.path.join(CURRENT_DIR, '../resources/shutterstock-cat--250.jpg')
    image_bw = cv.imread(image_path, 0)
    cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
    cv.imshow('image', image_bw)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print(f"image shape {image_bw.shape}")
    print(f"image size {image_bw.size}")

    image = cv.imread(image_path, 1)
    cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print(f"image shape {image.shape}")
    print(f"image size {image.size}")

    new_image = image
    new_image[23:55, 23:55] = (0, 255, 0)
    cv.imshow('image', new_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print(f"image shape {new_image.shape}")
    print(f"image size {new_image.size}")
