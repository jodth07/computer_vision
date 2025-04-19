import cv2 as cv

if __name__ == '__main__':

    butterfly = cv.imread('../resources/butterfly.jpg', cv.IMREAD_COLOR)
    cv.imshow('Butterfly', butterfly)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    b = butterfly[:, :, 0]
    g = butterfly[:, :, 1]
    r = butterfly[:, :, 2]

    with_trans = cv.merge((b, g, r, g))
    cv.imwrite('../target/butterfly_with_transparency.png', with_trans)

    cv.imshow('Butterfly with transparency', with_trans)

    cv.moveWindow('Butterfly with transparency', 0, butterfly.shape[1])
    cv.waitKey(0)
    cv.destroyAllWindows()
