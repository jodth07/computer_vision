"""
Find on the internet (or use a camera to take) three different types of images:
an indoor scene, outdoor scenery, and a close-up scene of a single object.
Implement an adaptive thresholding scheme to segment the images as best as you can.
"""

import cv2
from matplotlib import pyplot as plt

def adaptive_thresholding(image_path):
    # Read the image
    image = cv2.imread(image_path, 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = 3
    sig = 0

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(gray_image, (k, k), sig)

    # Adaptive thresholding
    adaptive_thresh_mean = cv2.adaptiveThreshold(blurred_image, 255,
                                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)

    adaptive_thresh_gaussian = cv2.adaptiveThreshold(blurred_image, 255,
                                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY, 11, 2)

    # Display the results
    titles = ['Original Image', 'Gaussian Blurred Image', 'Adaptive Threshold Mean', 'Adaptive Threshold Gaussian']
    images = [gray_image, blurred_image, adaptive_thresh_mean, adaptive_thresh_gaussian]

    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')



