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

    def detect_faces(self, cascade_path, input_image: Optional[np.ndarray] = None):
        """
        detect the human faces in the gray scaled versions of the original images
        """
        process_image = input_image if input_image is not None else self.image
        gray_img = cv.cvtColor(process_image, cv.COLOR_BGR2GRAY)
        face_cascade = cv.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray_img, 1.20, 16, minSize=(80,80))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv.rectangle(process_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return process_image

    @staticmethod
    def extract_faces(input_image: np.ndarray, face_cascade_path: str) -> list:
        """
        Extract faces from the input image using a Haar Cascade classifier.
        """
        gray_img = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
        face_cascade = cv.CascadeClassifier(face_cascade_path)
        faces = face_cascade.detectMultiScale(gray_img, 1.20, 16, minSize=(80, 80))

        extracted_faces = []
        for (x, y, w, h) in faces:
            face_region = input_image[y:y + h, x:x + w]
            extracted_faces.append(face_region)

        return extracted_faces

    @staticmethod
    def align_faces(faces: list) -> list:
        """
        Align faces to ensure they are centered and scaled properly.
        This is a placeholder function; actual implementation may vary based on requirements.
        """
        aligned_faces = []
        for face in faces:
            # Placeholder for alignment logic
            # For now, we just append the original face
            aligned_faces.append(face)
        return aligned_faces

    @staticmethod
    def detect_eyes(input_image, cascade_path):
        """
        detect the human eyes in the gray scaled versions of the original images
        """

        gray_img = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
        eyes_cascade = cv.CascadeClassifier(cascade_path)
        eyes = eyes_cascade.detectMultiScale(gray_img, 1.06, 26, minSize=(20, 20))

        print(len(eyes))

        if len(eyes) > 0:
            for (x, y, w, h) in eyes:
                center = (int(x + w / 2), int(y + h / 2))
                radius = int(round((w + h) / 4) * 1)
                cv.circle(input_image, center, radius, (0, 255, 0), 3)

        return input_image

    @staticmethod
    def align_face_by_eyes(face_img: np.ndarray, eye_cascade_path: str) -> Optional[np.ndarray]:
        """
        Align a face image so the eyes are horizontal.
        """
        gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
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

        # ✅ Fix here: cast to float
        eye_center = (
            float((eye_center_1[0] + eye_center_2[0]) / 2),
            float((eye_center_1[1] + eye_center_2[1]) / 2)
        )
        rot_mat = cv.getRotationMatrix2D(eye_center, angle, scale=1.0)
        aligned_face = cv.warpAffine(face_img, rot_mat, (face_img.shape[1], face_img.shape[0]), flags=cv.INTER_LINEAR)

        return aligned_face

    @staticmethod
    def resize_face(face_img: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
        """
        Resize a face image to a standard size.
        """
        resized = cv.resize(face_img, size, interpolation=cv.INTER_AREA)
        return resized

    @staticmethod
    def blur_eyes(face_img: np.ndarray, eye_cascade_path: str) -> np.ndarray:
        """
        Detect and blur eyes in a face image.
        """
        gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
        eye_cascade = cv.CascadeClassifier(eye_cascade_path)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        for (x, y, w, h) in eyes:
            eye_roi = face_img[y:y+h, x:x+w]
            blurred_eye = cv.GaussianBlur(eye_roi, (31, 31), 0)
            face_img[y:y+h, x:x+w] = blurred_eye

        return face_img

    @staticmethod
    def save_face_image(face_img: np.ndarray, output_dir: str, index: int) -> str:
        """
        Save a face image to a specified directory with an indexed filename.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"face_{index}.jpg")
        cv.imwrite(output_path, face_img)
        print(f"Saved face to {output_path}")
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
        image_processor.show_image(window_name="Original Image")

        extracted_faces = image_processor.extract_faces(image_processor.image, fc_path)

        for idx, face in enumerate(extracted_faces):
            # Step 2: Align face using detected eyes
            aligned = image_processor.align_face_by_eyes(face, eye_path)
            if aligned is not None:
                resized = image_processor.resize_face(aligned)
                blurred = image_processor.blur_eyes(resized, eye_path)
                image_processor.save_face_image(blurred, output_dir="processed_faces", index=tracker)
                image_processor.show_image(window_name=f"Aligned Face {tracker}", input_image=blurred)
                tracker += 1
