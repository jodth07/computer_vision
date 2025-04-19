import cv2 as cv
import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    height = 480
    width = 640
    dimensions = 3 # or channels
    # max/min value from range 0-255 of intensity
    maxValue = 255
    minValue = 0

    # create grey image from numpy arrays
    greyValue = 127
    grey = np.ones((height, width, dimensions), np.uint8) * greyValue
    cv.imshow('grey', grey)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # create black image from numpy arrays
    blackValue = minValue
    black = np.ones((height, width, dimensions), np.uint8) * blackValue
    cv.imshow('black', black)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # create white image from numpy arrays
    whiteValue = 255 # (2**8 - 1) - Max value
    white = np.ones((height, width, dimensions), np.uint8) * whiteValue
    cv.imshow('white', white)
    cv.waitKey(0)
    cv.destroyAllWindows()

    red_image = white.copy()
    blueMin, greenMin, redMin = (0, 0, 0)
    red_image[:, :] = (blueMin, greenMin, 255)
    cv.imshow('red_image', red_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    blue = white.copy()
    blue[:, :] = (255, greenMin, redMin)
    cv.imshow('blue', blue)
    cv.waitKey(0)
    cv.destroyAllWindows()

    green = white.copy()
    green[:,:] = (blueMin, 255, redMin)
    cv.imshow('green', green)
    cv.waitKey(0)
    cv.destroyAllWindows()

    yellowish_green = white.copy()
    yellowish_green[:,:] = (126, 255, redMin)
    cv.imshow('yellowish_green', yellowish_green)
    cv.waitKey(0)
    cv.destroyAllWindows()

    orange = white.copy()
    orange[:,:] = (0, 255, 255)
    cv.imshow('orange', orange)
    cv.waitKey(0)
    cv.destroyAllWindows()



