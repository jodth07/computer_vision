import cv2 as cv
import os
import numpy as np
import requests
from typing import Optional

BASE_IMAGE_URL = "https://raw.githubusercontent.com/jodth07/computer_vision/develop"

class ImageProcessor:

    def __init__(self):
        self.image = None
        self.resources = set()

    def load_image_from_url(self, url: str) -> "ImageProcessor":

        print(f"Downloading Image from {url} to memory")
        response = requests.get(url)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        self.image = image
        return self

    def download_resource(self, url: str, output_dir: str = "target") -> str:

        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(url)
        output_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(output_path):
            print(f"Downloading {output_path}...")
            response = requests.get(url)
            with open(output_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {output_path}.")
        else:
            print(f"{output_path} already exists.")
        self.resources.add(output_path)
        return output_path

    def load_image_from_file(self, file_path: str) -> "ImageProcessor":
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

    def detect_faces(self, face_cascade_path: str, input_image: Optional[np.ndarray] = None, extract: bool = True, draw_face: bool = False) -> list:
        """
        Extract faces from the input image using a Haar Cascade classifier.
        """
        orig_image = input_image if input_image is not None else self.image
        gray_img = cv.cvtColor(orig_image, cv.COLOR_BGR2GRAY)
        face_cascade = cv.CascadeClassifier(face_cascade_path)
        faces = face_cascade.detectMultiScale(gray_img, 1.03, minNeighbors=70, minSize=(20, 20))

        if len(faces) > 0:
            if extract:
                new_extracted_faces = []
                for (x, y, w, h) in faces:
                    face_region = orig_image[y:y + h, x:x + w]
                    new_extracted_faces.append(face_region)

                return new_extracted_faces

            if draw_face:
                for (x, y, w, h) in faces:
                    cv.rectangle(orig_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                return [orig_image]
        return []


    def align_face_by_eyes(self, eye_cascade_path: str, input_image: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Align a face image so the eyes are horizontal.
        """
        original_image = input_image if input_image is not None else self.image
        gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
        eye_cascade = cv.CascadeClassifier(eye_cascade_path)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        if len(eyes) < 2:
            print("Less than 2 eyes detected; skipping alignment.")
            return None

        # Sort eyes by x-coordinate
        eyes = sorted(eyes, key=lambda x: x[0])
        eye_1, eye_2 = eyes[:2]

        (x1, y1, w1, h1) = eye_1
        (x2, y2, w2, h2) = eye_2

        eye_center_1 = (x1 + w1 // 2, y1 + h1 // 2)
        eye_center_2 = (x2 + w2 // 2, y2 + h2 // 2)

        dx = eye_center_2[0] - eye_center_1[0]
        dy = eye_center_2[1] - eye_center_1[1]
        angle = np.degrees(np.arctan2(dy, dx))

        eye_center = (
            float((eye_center_1[0] + eye_center_2[0]) / 2),
            float((eye_center_1[1] + eye_center_2[1]) / 2)
        )
        rot_mat = cv.getRotationMatrix2D(eye_center, angle, scale=1.0)
        aligned_face = cv.warpAffine(original_image, rot_mat, (original_image.shape[1], original_image.shape[0]), flags=cv.INTER_LINEAR)

        return aligned_face

    @staticmethod
    def resize_face(face_img: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
        """
        Resize a face image to a standard size.
        """
        return cv.resize(face_img, size, interpolation=cv.INTER_AREA)

    def blur_eyes(self, eye_cascade_path: str, input_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect and blur eyes in a face image.
        """
        original_image = input_image if input_image is not None else self.image
        gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
        eye_cascade = cv.CascadeClassifier(eye_cascade_path)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        print(f"Detected {len(eyes)} eyes for blurring.")

        for (x, y, w, h) in eyes:
            print(f"Eye detected at x: {x}, y: {y}, width: {w}, height: {h}")
            eye_roi = original_image[y:y+h, x:x+w]
            blurred_eye = cv.GaussianBlur(eye_roi, (31, 31), 0)
            original_image[y:y+h, x:x+w] = blurred_eye

        return original_image

    def save_image_to_file(self, file_name: str, input_image: Optional[np.ndarray] = None, output_dir: str = "target"):

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)

        image = input_image if input_image is not None else self.image
        if image is None:
            raise ValueError("No image loaded.")

        cv.imwrite(output_path, image)
        print(f"Image saved to {output_path}")

        return output_path


if __name__ == '__main__':
    tracker = 0
    resources = [
        f"{BASE_IMAGE_URL}/resources/haarcascade_frontalface_default.xml",
        f"{BASE_IMAGE_URL}/resources/haarcascade_eye.xml",
        f"{BASE_IMAGE_URL}/resources/8_123_jdtea.studio.jpg",
        f"{BASE_IMAGE_URL}/resources/8_far_away_standing.jpg",
        f"{BASE_IMAGE_URL}/resources/8_124_far_away.jpg",
        f"{BASE_IMAGE_URL}/resources/8_1.jpg",
    ]

    image_processor = ImageProcessor()
    fc_path = image_processor.download_resource(resources[0])
    eye_path = image_processor.download_resource(resources[1])

    images = resources[2:]
    for image_resource in images:
        image_processor.load_image_from_url(image_resource)
        extracted_faces = image_processor.detect_faces(fc_path)

        for idx, face in enumerate(extracted_faces):
            resized = image_processor.resize_face(face)
            image_processor.show_image(window_name=f"jkj Face {tracker}", input_image=resized)
            aligned = image_processor.align_face_by_eyes(eye_path, input_image=resized)
            if aligned is not None:
                blurred = image_processor.blur_eyes(eye_path, resized)
                image_processor.save_image_to_file(input_image=blurred, output_dir="target", file_name=f"face_{tracker}.jpg")
                image_processor.show_image(window_name=f"Aligned Face {tracker}", input_image=blurred)
                tracker += 1

        # image_processor.show_image(window_name="Original Image")
        image_processor.detect_faces(fc_path, draw_face=True, extract=False)
        image_processor.blur_eyes(eye_path)
        image_processor.show_image(window_name=f"Aligned Face new")
