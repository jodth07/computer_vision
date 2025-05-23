import cv2 as cv
import numpy as np


if __name__ == '__main__':
    # img = cv.imread("../resources/6_closeup_mg.png", 0)
    img = cv.imread("../resources/6_indoor_scene.png", 0)
    # img = cv.imread("../resources/6_outdoor_scene.png", 0)
    cv.imshow("Original BW", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    ret, thresh_basic = cv.threshold(img, 70, 255, cv.THRESH_BINARY)
    cv.imshow("Threshold Basic", thresh_basic)
    cv.waitKey(0)
    cv.destroyAllWindows()

    thresh_adapt = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY, 115, 1)

    cv.imshow("Threshold Adaptive", thresh_adapt)
    cv.waitKey(0)
    cv.destroyAllWindows()

    thresh_adapt2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY, 11, 1)

    cv.imshow("Threshold Adaptive", thresh_adapt2)
    cv.waitKey(0)
    cv.destroyAllWindows()