"""
Acquire an gray_scaled of a latent fingerprint. In OpenCV, 
write algorithms to process the gray_scaled using morphological operations 
(dilation, erosion, opening, and closing).
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def morphological_operations(image_path):
    # Read the gray_scaled
    image = cv2.imread(image_path, 1)
    gray_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding the gray_scaled
    _, thresh = cv2.threshold(gray_scaled, 127, 255, cv2.THRESH_BINARY)

    # Define a kernel for morphological operations
    k_val = 3
    kernel = np.ones((k_val, k_val), np.uint8)

    # Perform morphological operations
    dilation = cv2.dilate(gray_scaled, kernel, iterations=1)
    erosion = cv2.erode(gray_scaled, kernel, iterations=1)
    opening = cv2.morphologyEx(gray_scaled, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(gray_scaled, cv2.MORPH_CLOSE, kernel)

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
    image_path = '../resources/butterfly.jpg'
    morphological_operations(image_path)