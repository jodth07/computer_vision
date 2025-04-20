import cv2 as cv
import os
import requests

base = "../resources/"

def download_resources():
    """
    Download the resources from the GitHub repository.
    """
    global base

    resources = [
        "https://raw.githubusercontent.com/jodth07/computer_vision/main/resources/jd.jpg",
        "https://raw.githubusercontent.com/jodth07/computer_vision/main/resources/haarcascade_frontalface_default.xml",
        "https://raw.githubusercontent.com/jodth07/computer_vision/main/resources/haarcascade_eye.xml"
    ]

    for url in resources:
        filename = os.path.basename(url)
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            response = requests.get(url)
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {filename}.")
        else:
            print(f"{filename} already exists.")
    base = ""


if __name__ == '__main__':

    """
    take a picture of yourself facing the frontal.  
    draw on the image 
        a red bounding box for your eyes and 
        a green circle around your face.  
    tag the image with the text “this is me”.
    """

    jd_image = cv.imread(f"{base}jd.jpg", 1)
    jd_gray = cv.cvtColor(jd_image, cv.COLOR_BGR2GRAY)

    face_cascade_path = f"{base}haarcascade_frontalface_default.xml"
    face_cascade = cv.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(jd_gray, 1.10, 5, minSize=(60,60))
    print(len(faces))
    print(len(faces[0]))

    for (x, y, w, h) in faces:
        center = (int(x + w / 2), int(y + h / 2))
        radius = int(round((w + h) / 4) * 1.3)
        cv.circle(jd_image, center, radius, (0, 255, 0), 3)

    eyes_cascade_path = f"{base}haarcascade_eye.xml"
    eyes_cascade = cv.CascadeClassifier(eyes_cascade_path)
    eyes = eyes_cascade.detectMultiScale(jd_gray, 1.10, 5, minSize=(30, 30))

    for (x, y, w, h) in eyes:
        cv.rectangle(jd_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow("this is me", jd_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
