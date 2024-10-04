import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def load_images_from_directory(directory):
    images = {}
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Add more extensions if needed
            img_path = os.path.join(directory, filename)
            images[filename] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return images

def find_most_similar_image(input_image_path, predefined_digits_directory):
    # Load input image
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (28, 28))  # Resize to match digit images if needed

    # Load predefined digit images
    predefined_images = load_images_from_directory(predefined_digits_directory)

    best_similarity = -1
    most_similar_image = None
    most_similar_image_name = ""

    for name, img in predefined_images.items():
        img_resized = cv2.resize(img, (28, 28))  # Resize to match input image size if needed

        # Compute SSIM
        similarity = ssim(input_image, img_resized)

        if similarity > best_similarity:
            best_similarity = similarity
            most_similar_image = img
            most_similar_image_name = name

    return most_similar_image_name, most_similar_image

# Example usage
input_image_path = "predefined_digits/9.png"  # Specify the path to your input image
predefined_digits_directory = "predefined_digits"  # Specify the path to the directory

similar_image_name, similar_image = find_most_similar_image(input_image_path, predefined_digits_directory)

# Display the most similar image
if similar_image is not None:
    cv2.imshow(f'Most Similar Image: {similar_image_name}', similar_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No similar image found.")
