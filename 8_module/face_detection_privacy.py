"""
First, using the appropriate trained cascade classifier,
    write one algorithm to detect the human faces in the gray scaled versions of the original images.
        Put a red boundary box around the detected face in the image in order to see what region the classifier deemed as a human face.
        If expected results are not achieved on the unprocessed images,
        apply processing steps before implementing the classifier for optimal results.

After the faces have been successfully detected, you will want to process only the extracted faces before detecting and applying blurring to hide the eyes. Although the eye classifierLinks to an external site. is fairly accurate, it is important that all faces are centered, rotated, and scaled so that the eyes are perfectly aligned. If expected results are not achieved, implement more image processing for optimal eye recognition. Now, apply a blurring method to blur the eyes out in the extracted image.

Inspect your results and write a summary describing the techniques you used to detect and blur the eyes out of human faces in images. Reflect on the challenges you faced and how you overcame these challenges.  Furthermore, discuss in your summary, the accuracy of your results for all three images and techniques you used to improve the accuracy after each repeated experiment.
"""
from typing import Optional

import cv2 as cv
import os
import numpy as np
import requests

BASE_IMAGE_URL = "https://raw.githubusercontent.com/jodth07/computer_vision/develop"

class ImageProcessor:

    def __init__(self):
        self.image = None
        self.resources = set()

    def load_image_from_url(self, url: str):
        """
        Import image from url
        """

        print(f"Downloading Image from {url} to memory")
        response = requests.get(url)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        self.image = image
        return self

    def download_resource(self, ufl):
        """
        Download the resources from the GitHub repository.
        """
        filename = os.path.basename(ufl)
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            response = requests.get(ufl)
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {filename}.")
        else:
            print(f"{filename} already exists.")
        self.resources.add(filename)
        return filename

    def load_image_from_file(self, file_path: str):
        """
        Load an image from a file path.
        """
        print(f"Loading Image from {file_path} to memory")

        self.image = cv.imread(file_path)
        if self.image is None:
            raise ValueError(f"Could not read image from {file_path}")
        print(f"Image loaded from {file_path}")
        return self

    def show_image(
            self, window_name: str = "Image", input_image: Optional[np.ndarray] = None
    ) -> "ImageProcessor":
        """
        Display the loaded image in a window.
        """
        print(f"Previewing Image in {window_name}")

        image = input_image if input_image is not None else self.image
        if image is None:
            raise ValueError("No image loaded.")
        print(f"Press `q` to continue")

        cv.imshow(window_name, image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return self

    def save_image_to_file(self, file_path: str):
        """
        Write a copy of the image to any directory
        """
        if self.image is None:
            raise ValueError("No image loaded.")
        cv.imwrite(file_path, self.image)
        print(f"Image saved to {file_path}")
        return self

def detect_faces_and_eyes():
    """
    take a picture of yourself facing the frontal.
    draw on the image
        a red bounding box for your eyes and
        a green circle around your face.
    tag the image with the text “this is me”.
    """

    resources = [
        "{BASE_IMAGE_URL}/resources/jd.jpg",
        "{BASE_IMAGE_URL}/resources/haarcascade_frontalface_default.xml",
        "{BASE_IMAGE_URL}/resources/haarcascade_eye.xml"
    ]

    image_processor = ImageProcessor()
    image_processor.load_image_from_url(resources[0])
    image_processor.show_image(window_name="Original Image")

    jd_image = image_processor.image
    jd_gray = cv.cvtColor(jd_image, cv.COLOR_BGR2GRAY)

    face_cascade_path = image_processor.download_resource(resources[1])
    eyes_cascade_path = image_processor.download_resource(resources[2])

    face_cascade = cv.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(jd_gray, 1.10, 5, minSize=(60,60))
    print(len(faces))
    print(len(faces[0]))

    for (x, y, w, h) in faces:
        center = (int(x + w / 2), int(y + h / 2))
        radius = int(round((w + h) / 4) * 1.3)
        cv.circle(jd_image, center, radius, (0, 255, 0), 3)

    eyes_cascade = cv.CascadeClassifier(eyes_cascade_path)
    eyes = eyes_cascade.detectMultiScale(jd_gray, 1.10, 5, minSize=(40, 40))

    for (x, y, w, h) in eyes:
        cv.rectangle(jd_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow("this is me", jd_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def detect_face(input_image, cascade_path):
    """
    detect the human faces in the gray scaled versions of the original images
    """

    gray_img = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray_img, 1.50, 12, minSize=(60,60))
    
    print(len(faces))
    # print(len(faces[0]))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv.rectangle(input_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return input_image


def detect_eyes(input_image, cascade_path):
    """
    detect the human eyes in the gray scaled versions of the original images
    """

    gray_img = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    eyes_cascade = cv.CascadeClassifier(cascade_path)
    eyes = eyes_cascade.detectMultiScale(gray_img, 1.25, 6, minSize=(30, 30))

    print(len(eyes))
    # print(len(eyes[0]))

    # for (x, y, w, h) in eyes:
    #     cv.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(eyes) > 0:
        for (x, y, w, h) in eyes:
            center = (int(x + w / 2), int(y + h / 2))
            radius = int(round((w + h) / 4) * 1)
            cv.circle(input_image, center, radius, (0, 255, 0), 1)

    return input_image


if __name__ == '__main__':
    resources = [
        f"{BASE_IMAGE_URL}/resources/haarcascade_frontalface_default.xml",
        f"{BASE_IMAGE_URL}/resources/haarcascade_eye.xml",
        f"{BASE_IMAGE_URL}/resources/8_123_jdtea.studio.jpg",
        f"{BASE_IMAGE_URL}/resources/8_1.jpg",
    ]

    image_processor = ImageProcessor()
    fc_path = image_processor.download_resource(resources[0])
    eye_path = image_processor.download_resource(resources[1])

    # image_processor.load_image_from_url(resources[2])
    image_processor.load_image_from_url(resources[3])
    image_processor.show_image(window_name="Original Image")

    # image = image_processor.image
    new_image = detect_face(image_processor.image, fc_path)
    # image_processor.show_image(input_image=new_image)
    image_processor.show_image(window_name="Original Image2")
    # image_processor.show_image(window_name="Original Image")

    with_eyes = detect_eyes(new_image, eye_path)
    image_processor.show_image(input_image=with_eyes, window_name="Original Image3")

    # detect_faces_and_eyes()
