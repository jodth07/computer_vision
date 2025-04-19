import numpy as np
import cv2 as cv

if __name__ == '__main__':
    color = cv.imread('../resources/butterfly.jpg', cv.IMREAD_COLOR)
    # cv.imshow('Butterfly', color)
    # cv.moveWindow('Butterfly', 0, 0)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    print(color.shape)
    height, width, channels = color.shape

    b, g, r = cv.split(color)
    rgb_split = np.empty([height, width * 3, channels], dtype=np.uint8)
    rgb_split[:, 0:width] = cv.merge([b, b, b])
    rgb_split[:, width:width * 2] = cv.merge([g, g, g])
    rgb_split[:, width * 2:width * 3] = cv.merge([r, r, r])
    print(f"rgb_split.shape {rgb_split.shape}")
    cv.imshow('Channels', rgb_split)
    cv.moveWindow('Channels', 0, height)

    hsv = cv.cvtColor(color, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    hsv_split = np.concatenate((h, s, v), axis=1)
    # hsv_split[:, 0:width] = cv.merge([h, h, h])
    # hsv_split[:, width:width * 2] = cv.merge([s, s, s])
    # hsv_split[:, width * 2:width * 3] = cv.merge([v, v, v])
    print(f"hsv_split.shape: {hsv_split.shape}")
    cv.imshow('Frequency', hsv_split)
    cv.moveWindow('Frequency', 0, 0)


    cv.waitKey(0)
    cv.destroyAllWindows()

    # cv.imshow('Butterfly', color)

