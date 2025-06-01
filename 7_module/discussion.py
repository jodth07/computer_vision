import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def create_synthetic_image(height, width, square_top_left, circle_center, square_size, circle_radius):
    image = np.ones((height, width, 3), np.uint8) * 127
    square_bottom_right = (square_top_left[0] + square_size, square_top_left[1] + square_size)
    cv.rectangle(image, square_top_left, square_bottom_right, (0, 155, 255), -1)
    cv.circle(image, circle_center, circle_radius, (0, 255, 0), -1)
    return image

def apply_canny(image, threshold1, threshold2):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.Canny(gray, threshold1, threshold2)

def apply_sobel(img: np.ndarray) -> np.ndarray:
    # Denoising step
    denoised = cv.medianBlur(img, 3)
    denoised = cv.GaussianBlur(denoised, (3, 3), 0)

    # Convert to grayscale
    gray = cv.cvtColor(denoised, cv.COLOR_BGR2GRAY)

    # Apply Sobel filter in both x and y directions
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    sobel = cv.magnitude(sobelx, sobely)
    sobel = cv.convertScaleAbs(sobel)

    return sobel

def apply_laplacian(img: np.ndarray) -> np.ndarray:
    # Denoising step
    denoised = cv.medianBlur(img, 3)
    denoised = cv.GaussianBlur(denoised, (3, 3), 0)

    # Convert to grayscale
    gray = cv.cvtColor(denoised, cv.COLOR_BGR2GRAY)

    # Apply Laplacian
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    abs_laplacian = cv.convertScaleAbs(laplacian)

    return abs_laplacian

def generate_ground_truth(height, width, square_top_left, circle_center, square_size, circle_radius):
    ground_truth = np.zeros((height, width), np.uint8)
    square_bottom_right = (square_top_left[0] + square_size, square_top_left[1] + square_size)
    cv.rectangle(ground_truth, square_top_left, square_bottom_right, 255, 1)
    cv.circle(ground_truth, circle_center, circle_radius, 255, 1)
    return ground_truth

def evaluate_edge_detection(ground_truth, detected_edges):
    ground_truth_binary = (ground_truth > 0).astype(np.uint8)
    detected_edges_binary = (detected_edges > 0).astype(np.uint8)
    true_positives = np.sum((ground_truth_binary == 1) & (detected_edges_binary == 1))
    false_positives = np.sum((ground_truth_binary == 0) & (detected_edges_binary == 1))
    false_negatives = np.sum((ground_truth_binary == 1) & (detected_edges_binary == 0))
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"Precision": precision, "Recall": recall, "F1-Score": f1_score}

def add_noise(image, noise_type='gaussian', mean=0, var=10):
    if noise_type == 'gaussian':
        gaussian_noise = np.random.normal(mean, var ** 0.5, image.shape).astype(np.int16)
        noisy_image = image.astype(np.int16) + gaussian_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        noisy_image = image.copy()
        salt_prob, pepper_prob = 0.02, 0.02
        num_salt = int(salt_prob * image.size)
        num_pepper = int(pepper_prob * image.size)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 255
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 0
        return noisy_image
    else:
        raise ValueError("Unsupported noise type.")

def display_images_with_scores(images, titles, scores, save_path=None):
    num_images = len(images)
    cols = 3
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    if rows == 1: axes = [axes]

    for i, ax in enumerate(np.ravel(axes)):
        if i < num_images:
            img = images[i]
            if len(img.shape) == 2:  # grayscale
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            ax.set_title(f"{titles[i]}\n{scores[i]}", fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved visualization to {save_path}")
    plt.show()

def run_experiment(output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    HEIGHT = 480
    WIDTH = 640
    RADIUS = 100
    SQUARE_SIZE = 180
    c_center = (3 * WIDTH // 4, HEIGHT // 2)
    s_top_left = (WIDTH // 4, HEIGHT // 4)

    synthetic_image = create_synthetic_image(HEIGHT, WIDTH, s_top_left, c_center, SQUARE_SIZE, RADIUS)
    ground_truth_edges = generate_ground_truth(HEIGHT, WIDTH, s_top_left, c_center, SQUARE_SIZE, RADIUS)

    noise_variants = {
        "Clean": synthetic_image,
        "Gaussian Noise": add_noise(synthetic_image, 'gaussian', var=20),
        "Salt-Pepper Noise": add_noise(synthetic_image, 'salt_pepper'),
        "Combined Noise": add_noise(add_noise(synthetic_image, 'gaussian', var=20), 'salt_pepper'),
    }

    thresholds = [(50, 150), (100, 200)]

    for noise_name, noisy_image in noise_variants.items():
        images = [noisy_image]  # Include original noisy image
        titles = [f"Original - {noise_name}"]
        scores = [""]

        for t1, t2 in thresholds:
            edges = apply_canny(noisy_image, t1, t2)
            metrics = evaluate_edge_detection(ground_truth_edges, edges)
            images.append(edges)
            img_title = f"Canny ({t1},{t2})"
            titles.append(img_title)
            img_score = f"P: {metrics['Precision']:.2f}, R: {metrics['Recall']:.2f}, F1: {metrics['F1-Score']:.2f}"
            scores.append(img_score)
            print(img_title)
            print(img_score)

        sobel_edges = apply_sobel(noisy_image)
        metrics = evaluate_edge_detection(ground_truth_edges, sobel_edges)
        images.append(sobel_edges)
        img_title = "Sobel"
        titles.append(img_title)
        img_score = f"P: {metrics['Precision']:.2f}, R: {metrics['Recall']:.2f}, F1: {metrics['F1-Score']:.2f}"
        scores.append(img_score)
        print(img_title)
        print(img_score)

        laplacian_edges = apply_laplacian(noisy_image)
        metrics = evaluate_edge_detection(ground_truth_edges, laplacian_edges)
        images.append(laplacian_edges)
        img_title = "Laplacian"
        titles.append(img_title)
        img_score = f"P: {metrics['Precision']:.2f}, R: {metrics['Recall']:.2f}, F1: {metrics['F1-Score']:.2f}"
        scores.append(img_score)
        print(img_title)
        print(img_score)

        save_path = os.path.join(output_dir, f"comparison_{noise_name.replace(' ', '_')}.png")
        display_images_with_scores(images, titles, scores, save_path=save_path)

if __name__ == '__main__':
    run_experiment()
