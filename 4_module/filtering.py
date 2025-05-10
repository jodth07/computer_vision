import numpy as np

import cv2


def to_plt_rgb(image):
    """
    Convert an image from BGR to RGB format for display in matplotlib.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


import matplotlib.pyplot as plt


# def mean_blur():


if __name__ == "__main__":

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
    # fig, (ax1, ax4, ax5, ax6) = plt.subplots(nrows = 1, ncols = 4, figsize=(15, 5))

    img = cv2.imread("../resources/portrait.jpg")
    ax1.imshow(to_plt_rgb(img))
    ax1.set_title("Original")
    #
    median_blur = cv2.medianBlur(img, 5)
    ax2.imshow(to_plt_rgb(median_blur))
    ax2.set_title("Median - 5x5 Kernel")

    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 3)
    ax3.imshow(to_plt_rgb(gaussian_blur))
    ax3.set_title("Gaussian - 5x5 Kernel")

    laplacian_blur = cv2.Laplacian(img, cv2.CV_32F)
    ax4.imshow(cv2.cvtColor(laplacian_blur, cv2.COLOR_BGR2RGB))
    ax4.set_title("Laplacian Operator Applied")

    # blur = cv2.GaussianBlur(img, (7, 7), 10)

    gray = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2GRAY)
    laplacian_g_blur = cv2.Laplacian(gray, cv2.CV_32F)
    ax5.imshow(cv2.cvtColor(laplacian_g_blur, cv2.COLOR_BGR2RGB))
    ax5.set_title("Laplacian Applied After Blurring")

    # gaussian_blur_7 = cv2.GaussianBlur(img, (7,7), 10)
    # gray = cv2.cvtColor(gaussian_blur_7, cv2.COLOR_BGR2GRAY)
    # laplacian_g_blur = cv2.Laplacian(gray, cv2.CV_32F)
    # ax6.imshow(cv2.cvtColor(laplacian_g_blur, cv2.COLOR_BGR2RGB))
    # ax6.set_title('Laplacian Applied After Blurring 7/110')

    plt.show()
