import cv2
import os
import time
import numpy as np

from extract import extract_frames

from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from skimage.metrics import structural_similarity as ssim


def load_images_from_directory(directory):
    images = {}
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Add more extensions if needed
            img_path = os.path.join(directory, filename)
            images[filename] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return images

def find_most_similar_image(input_image, predefined_digits_directory):
    # Load input image

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


# Load the ResNet50 model pre-trained on ImageNet and exclude the final layers
def load_model():
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)  # Using 'avg_pool' as embedding layer
    return model

# Function to preprocess an image for ResNet50
def preprocess_image(image):
    # Convert grayscale image to 3-channel image (RGB)
    if len(image.shape) == 2:  # If the image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels
    
    image = cv2.resize(image, (224, 224))  # Resize to ResNet50 expected size
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Preprocess for ResNet50
    return image

# Function to generate embeddings from a batch of images using ResNet50
def generate_embeddings(model, images):
    preprocessed_images = np.vstack([preprocess_image(img) for img in images])
    embeddings = model.predict(preprocessed_images)
    return embeddings

# Function to slice the input image into individual digits
def slice_image(image, num_slices):
    height, width = image.shape
    slice_width = width // num_slices
    slices = []
    for i in range(num_slices):
        if i==4:
            digit_slice = image[:, i * slice_width+1:(i + 2) * slice_width+1]
        else:    
            digit_slice = image[:, i * slice_width:(i + 1) * slice_width]
        slices.append(digit_slice)
    return slices

# Function to load the predefined digit images and generate their embeddings
def load_predefined_digits(predefined_dir, model):
    digits_embeddings = {}
    for i in range(10):
        digit_path = os.path.join(predefined_dir, f'{i}.png')
        digit_image = cv2.imread(digit_path, cv2.IMREAD_GRAYSCALE)
        embedding = generate_embeddings(model, [digit_image])[0]  # Get embedding for the digit image
        digits_embeddings[i] = embedding
    return digits_embeddings

# Function to calculate cosine similarity between two embeddings
def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

# Function to recognize the digit based on cosine similarity with predefined embeddings
def recognize_digit(digit_slice, predefined_embeddings, model):
    digit_embedding = generate_embeddings(model, [digit_slice])[0]
    best_match = None
    highest_similarity = -1
    
    # Compare against all predefined digit embeddings (0-9)
    for digit, predefined_embedding in predefined_embeddings.items():
        similarity = calculate_cosine_similarity(digit_embedding, predefined_embedding)
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = digit
    
    return best_match

# Main function to process the image and recognize digits using ResNet50
def process_image(input_image_path, predefined_embeddings, model):
    # Load input image
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Slice the input image into individual digits (change num_slices as necessary)
    num_slices = 5  # Adjust based on the number of digits in the image
    digit_slices = slice_image(input_image, num_slices)

    # Recognize each digit slice
    detected_number = []
    for digit_slice in digit_slices:

        predefined_digits_directory = "predefined_digits"  # Specify the path to the directory

        similar_image_name, similar_image = find_most_similar_image(digit_slice, predefined_digits_directory)

        # cv2.imshow("image",similar_image)
        # cv2.waitKey(0)
        # recognized_digit = recognize_digit(digit_slice, predefined_embeddings, model)
        recognized_digit = recognize_digit(similar_image, predefined_embeddings, model)
        detected_number.append(recognized_digit)

    # Join the recognized digits into a final number
    final_number = ''.join(map(str, detected_number))
    return final_number

# Function to iterate through all images in the extracted_images directory
def process_all_images_in_directory(directory, predefined_digits_dir, model):
    final_outputs = {}
    
    # Create logs directory if it doesn't exist
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)

    predefined_embeddings = load_predefined_digits(predefined_digits_dir, model)

    # Open log file for writing
    log_file_path = os.path.join(logs_dir, f'output_log_{time.time()}.txt')
    with open(log_file_path, 'w') as log_file:
        # Loop through each file in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_path = os.path.join(directory, filename)
                print(f"Processing image: {filename}")
                
                # Process the image to recognize the digits
                final_number = process_image(image_path, predefined_embeddings, model)
                
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
            for j in range(1,i-val):
                missing_frames.append(val+j)
        val = i

    print(missing_frames)
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Missing Frames: {missing_frames}")
        

if __name__ == "__main__":
    # Load ResNet50 model
    model = load_model()
    s = time.time()
    
    # Define paths
    extracted_images_dir = 'extracted_frames'  # Directory with input images
    predefined_digits_dir = 'predefined_digits'  # Directory with predefined digit images
    video_link = 'assets/video.mp4'
    
    extract_frames(video_link)
    # Process all images in the directory and recognize digits
    process_all_images_in_directory(extracted_images_dir, predefined_digits_dir, model)
    print(f"Total processing time: {time.time() - s:.2f} seconds")
    cv2.destroyAllWindows()
