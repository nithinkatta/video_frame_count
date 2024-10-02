import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import time

# Load the input image (the one with digits)
input_image_path = 'extracted_frames/frame_1.png'  # Replace with your image path
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Path to predefined digit images (0-9)
predefined_digits_dir = 'predefined_digits'  # This folder should contain images '0.png', '1.png', ..., '9.png'

# Function to slice the input image into individual digits
def slice_image(image, num_slices):
    height, width = image.shape
    print(f"Image Dimensions: Width={width}, Height={height}")  # Debugging info
    
    slice_width = width // num_slices  # Calculate the width of each slice
    print(f"Slice Width: {slice_width}")  # Debugging info
    
    slices = []
    for i in range(num_slices):
        digit_slice = image[:, i * slice_width:(i + 1) * slice_width]
        slices.append(digit_slice)
        # Show each slice for debugging
        cv2.imshow(f"Digit Slice {i+1}", digit_slice)
        cv2.waitKey(0)
    return slices

# Function to compute SSIM similarity between two images
def compare_images(image1, image2):
    resized_image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))  # Resize to match dimensions
    score, _ = compare_ssim(resized_image1, image2, full=True)
    return score

# Function to find the best matching digit for each slice
def recognize_digit(digit_slice, predefined_digits):
    best_match = None
    highest_score = -1
    for digit, digit_image in predefined_digits.items():
        score = compare_images(digit_slice, digit_image)
        if score > highest_score:
            highest_score = score
            best_match = digit
    return best_match

# Load the predefined digit images (0-9)
def load_predefined_digits(predefined_dir):
    digits = {}
    for i in range(10):
        digit_path = os.path.join(predefined_dir, f'{i}.png')
        digit_image = cv2.imread(digit_path, cv2.IMREAD_GRAYSCALE)
        digits[i] = digit_image
    return digits

# Main function to process the image and extract digits
def process_image(input_image_path, predefined_digits_dir):
    # Load input image
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Debugging: Show the full input image to confirm it's loaded correctly
    cv2.imshow("Input Image", input_image)
    cv2.waitKey(0)
    
    # Slice the input image into individual digits (update num_slices as necessary)
    num_slices = 5  # Change this based on the actual number of digits
    digit_slices = slice_image(input_image, num_slices)
    
    # Load predefined digit images (0-9)
    predefined_digits = load_predefined_digits(predefined_digits_dir)

    # Recognize each digit slice
    detected_number = []
    for digit_slice in digit_slices:
        recognized_digit = recognize_digit(digit_slice, predefined_digits)
        detected_number.append(recognized_digit)

    # Join the recognized digits into a final number
    final_number = ''.join(map(str, detected_number))
    return final_number

# Example usage
if __name__ == "__main__":
    final_number = process_image(input_image_path, predefined_digits_dir)
    print(f"The detected number is: {final_number}")
    cv2.destroyAllWindows()
