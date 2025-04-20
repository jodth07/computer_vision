import cv2 as cv

if __name__ == '__main__':

    image = cv.imread("../resources/faces.jpeg", 1)
    cv.imshow("image", image)
    cv.waitKey(0)

    image2 = image.copy()

    # cv.imshow("image", image)
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("grey", grey)
    cv.waitKey(0)
    cascade_path = "../resources/haarcascade_frontalface_default.xml"

    face_cascade = cv.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(grey, 1.10, 5, minSize=(40,40))
    print(len(faces))
    print(len(faces[0]))

    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("image2", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow("image3", image2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    eye_cascade_path = "../resources/haarcascade_eye.xml"
    eyes_cascade = cv.CascadeClassifier(eye_cascade_path)
    eyes = eyes_cascade.detectMultiScale(grey, 1.05, 5, minSize=(20, 20))
    print(len(eyes))
    print(len(eyes[0]))

    for (x, y, w, h) in eyes:
        center = (int(x + w / 2), int(y + h / 2))
        radius = int(round((w + h) / 4))
        cv.circle(image2, center, radius, (255, 0, 0), 2)

    cv.imshow("eyes", image2)
    cv.waitKey(0)
    cv.destroyAllWindows()