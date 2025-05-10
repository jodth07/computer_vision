import cv2 as cv
import numpy as np

import cv2
import numpy as np

def apply_gabor_filter(gray, kernel_size=31, sigma=8.0, theta=np.pi/4, lambd_=10.0, gamma=0.5):
    g_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd_, gamma, 0, ktype=cv2.CV_32F)
    return cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)

def apply_adaptive_threshold(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def compute_fourier_spectrum(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)  # Avoid log(0)
    return np.uint8(np.clip(magnitude, 0, 255))

def compute_lbp(gray):
    lbp = np.zeros_like(gray)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i, j]
            binary = [gray[i-1, j-1] >= center, gray[i-1, j] >= center, gray[i-1, j+1] >= center,
                      gray[i, j+1] >= center, gray[i+1, j+1] >= center, gray[i+1, j] >= center,
                      gray[i+1, j-1] >= center, gray[i, j-1] >= center]
            lbp[i, j] = sum([bit << idx for idx, bit in enumerate(binary)])
    return lbp


if __name__ == '__main__':
    # load the image
    image = cv.imread('../resources/bank_notes.jpg')
    if image is None:
        raise ValueError("Could not read image from bank_notes.jpg")
    height, width, channels = image.shape

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    equalized = cv.equalizeHist(grey)

    # edges = cv.Canny(grey, 50, 100)
    gabor_filter = apply_gabor_filter(grey, 5)
    gabor_filter2 = apply_gabor_filter(grey, 15)
    gabor_filter3 = apply_gabor_filter(grey, 25)
    gabor_filter4 = apply_gabor_filter(grey, 35)

    cv.imshow('gabor_filter', gabor_filter)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('gabor_filter2', gabor_filter2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('gabor_filter3', gabor_filter3)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('gabor_filter4', gabor_filter4)
    cv.waitKey(0)
    cv.destroyAllWindows()



    gabor = apply_gabor_filter(grey)
    adaptive_thresh = apply_adaptive_threshold(grey)

    h, s, v = cv.split(hsv)

    hsv_split = np.ones([height, width * 10, channels], dtype=np.uint8)
    hsv_split[:, 0:width] = cv.merge([grey, grey, grey])  # Convert to 3 channels for display
    hsv_split[:, width:width * 2] = cv.merge([equalized, equalized, equalized])
    hsv_split[:, width * 2:width * 3] = cv.merge([edges, edges, edges])
    hsv_split[:, width * 3:width * 4] = cv.merge([h, h, h])
    hsv_split[:, width * 4:width * 5] = cv.merge([s, s, s])
    hsv_split[:, width * 5:width * 6] = cv.merge([v, v, v])
    hsv_split[:, width * 6:width * 7] = hsv
    hsv_split[:, width * 7:width * 8] = image
    hsv_split[:, width * 8:width * 9] = cv.merge([gabor, gabor, gabor])  # Convert to 3 channels for display
    hsv_split[:, width * 9:width * 10] = cv.merge([adaptive_thresh, adaptive_thresh, adaptive_thresh])

    # Add labels
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    white = (255, 255, 255)  # White text
    black = (0, 0, 0)
    thickness = 1
    bpos = height - 3

    cv.putText(hsv_split, 'Grey', (5, bpos), font, font_scale, black, thickness)
    cv.putText(hsv_split, 'Equalized', (width + 5, bpos), font, font_scale, black, thickness)
    cv.putText(hsv_split, 'Edges', (width * 2 + 5, bpos), font, font_scale, white, thickness)
    cv.putText(hsv_split, 'Hue', (width * 3 + 5, bpos), font, font_scale, white, thickness)
    cv.putText(hsv_split, 'Saturation', (width * 4 + 5, bpos), font, font_scale, white, thickness)
    cv.putText(hsv_split, 'Value', (width * 5 + 5, bpos), font, font_scale, black, thickness)
    cv.putText(hsv_split, 'HSV', (width * 6 + 5, bpos), font, font_scale, black, thickness)
    cv.putText(hsv_split, 'Original', (width * 7 + 5, bpos), font, font_scale, black, thickness)
    cv.putText(hsv_split, 'gabor', (width * 8 + 5, bpos), font, font_scale, black, thickness)
    cv.putText(hsv_split, 'Adap Thresh', (width * 9 + 5, bpos), font, font_scale, black, thickness)

    cv.imshow('hsv_split', hsv_split)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite('../resources/bank_notes_updated.jpg', hsv_split)
