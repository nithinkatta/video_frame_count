import cv2
import os
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
from extract import *

# Load images from a directory (grayscale)
def load_images_from_directory(directory):
    images = {}
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Add more extensions if needed
            img_path = os.path.join(directory, filename)
            images[filename.split('.')[0]] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return images

# Function to slice the input image into individual digits
def slice_image(image, num_slices):
    height, width = image.shape
    slice_width = width // num_slices
    slices = []
    for i in range(num_slices):
        if i == 4:
            digit_slice = image[:, i * slice_width + 1:(i + 2) * slice_width + 1]
        else:
            digit_slice = image[:, i * slice_width:(i + 1) * slice_width]
        slices.append(digit_slice)
    return slices

# SSIM-based Pixel Comparison for recognizing digits
def recognize_digit_ssim(digit_slice, predefined_digits_directory):
    # Load predefined digit images (0-9)
    predefined_images = load_images_from_directory(predefined_digits_directory)
    
    best_match = None
    highest_similarity = -1
    
    for digit, predefined_image in predefined_images.items():
        # Resize the predefined digit to match the size of the input digit slice
        predefined_resized = cv2.resize(predefined_image, (digit_slice.shape[1], digit_slice.shape[0]))

        # Compute SSIM
        similarity = ssim(digit_slice, predefined_resized)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = digit

    return int(best_match)

# MSE calculation between two images
def mse(imageA, imageB):
    # Mean Squared Error between the two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# MSE-based Pixel Comparison for recognizing digits
def recognize_digit_mse(digit_slice, predefined_digits_directory):
    # Load predefined digit images (0-9)
    predefined_images = load_images_from_directory(predefined_digits_directory)
    
    best_match = None
    lowest_error = float("inf")
    
    for digit, predefined_image in predefined_images.items():
        # Resize the predefined digit to match the size of the input digit slice
        predefined_resized = cv2.resize(predefined_image, (digit_slice.shape[1], digit_slice.shape[0]))

        # Compute MSE
        error = mse(digit_slice, predefined_resized)

        if error < lowest_error:
            lowest_error = error
            best_match = digit

    return int(best_match)

# Main function to process the image and recognize digits using SSIM or MSE
def process_image(input_image_path, predefined_digits_directory, use_ssim=True):
    # Load input image
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Slice the input image into individual digits (adjust num_slices as necessary)
    num_slices = 5  # Adjust based on the number of digits in the image
    digit_slices = slice_image(input_image, num_slices)

    # Recognize each digit slice using SSIM or MSE
    detected_number = []
    for digit_slice in digit_slices:
        if use_ssim:
            recognized_digit = recognize_digit_ssim(digit_slice, predefined_digits_directory)
        else:
            recognized_digit = recognize_digit_mse(digit_slice, predefined_digits_directory)
        detected_number.append(recognized_digit)

    # Join the recognized digits into a final number
    final_number = ''.join(map(str, detected_number))
    return final_number

# Function to iterate through all images in the directory
def process_all_images_in_directory(directory, predefined_digits_dir, use_ssim=True):
    final_outputs = {}
    
    # Create logs directory if it doesn't exist
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)

    # Open log file for writing
    log_file_path = os.path.join(logs_dir, f'output_log_{time.time()}.txt')
    with open(log_file_path, 'w') as log_file:
        # Loop through each file in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_path = os.path.join(directory, filename)
                print(f"Processing image: {filename}")
                
                # Process the image to recognize the digits
                final_number = process_image(image_path, predefined_digits_dir, use_ssim)
                
                # Store the result for each image
                final_outputs[filename] = final_number
                
                # Log the result to the file
                log_file.write(f"Image: {filename}, Detected Number: {final_number}\n")

    frames_found = []

    # Print the final output for all images after processing
    print("\nFinal output for all images:")
    for filename, detected_number in final_outputs.items():
        print(f"Image: {filename}, Detected Number: {detected_number}")
        frames_found.append(int(detected_number))
    
    frames_found.sort()
    missing_frames = []
    val = frames_found[0]
    for i in frames_found[1:]:
        if i - val > 1:
            for j in range(1, i - val):
                missing_frames.append(val + j)
        val = i

    print("Missing Frames:", missing_frames)
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Missing Frames: {missing_frames}")

if __name__ == "__main__":
    s = time.time()
    
    # Define paths
    extracted_images_dir = 'extracted_frames'  # Directory with input images
    predefined_digits_dir = 'predefined_digits'  # Directory with predefined digit images
    video_link = 'assets/video.mp4'

    extract_frames(video_link)
    # Process all images in the directory and recognize digits using SSIM or MSE
    process_all_images_in_directory(extracted_images_dir, predefined_digits_dir, use_ssim=True)  # Use SSIM
    # process_all_images_in_directory(extracted_images_dir, predefined_digits_dir, use_ssim=False)  # Use MSE
    
    print(f"Total processing time: {time.time() - s:.2f} seconds")
    cv2.destroyAllWindows()
