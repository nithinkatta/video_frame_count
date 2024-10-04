import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input

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

# Function to generate embeddings from an image using ResNet50
def generate_embedding(model, image):
    preprocessed_image = preprocess_image(image)
    embedding = model.predict(preprocessed_image)
    return embedding

# Function to slice the input image into individual digits
def slice_image(image, num_slices):
    height, width = image.shape
    slice_width = width // num_slices
    slices = []
    for i in range(num_slices):
        digit_slice = image[:, i * slice_width:(i + 1) * slice_width]
        cv2.imshow('digit_slices', digit_slice)
        cv2.waitKey(0)
        
        slices.append(digit_slice)
    return slices

# Function to load the predefined digit images and generate their embeddings
def load_predefined_digits(predefined_dir, model):
    digits_embeddings = {}
    for i in range(10):
        digit_path = os.path.join(predefined_dir, f'{i}.png')
        digit_image = cv2.imread(digit_path, cv2.IMREAD_GRAYSCALE)
        embedding = generate_embedding(model, digit_image)
        digits_embeddings[i] = embedding
    return digits_embeddings

# Function to calculate cosine similarity between two embeddings
def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)[0][0]

# Function to recognize the digit based on cosine similarity with predefined embeddings
def recognize_digit(digit_slice, predefined_embeddings, model):
    digit_embedding = generate_embedding(model, digit_slice)
    best_match = None
    highest_similarity = -1
    
    # Compare against all predefined digit embeddings (0-9)
    for digit, predefined_embedding in predefined_embeddings.items():
        similarity = calculate_cosine_similarity(digit_embedding, predefined_embedding)
        print(f"Digit {digit}: Cosine Similarity = {similarity}")  # Debugging info
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = digit
    
    print(f"Best Match: {best_match} with Similarity: {highest_similarity}")
    return best_match

# Main function to process the image and recognize digits using ResNet50
def process_image(input_image_path, predefined_digits_dir, model):
    # Load input image
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Slice the input image into individual digits (change num_slices as necessary)
    num_slices = 5  # Adjust based on the number of digits in the image
    digit_slices = slice_image(input_image, num_slices)

    # Load predefined digit images and their embeddings
    predefined_embeddings = load_predefined_digits(predefined_digits_dir, model)

    # Recognize each digit slice
    detected_number = []
    for digit_slice in digit_slices:
        recognized_digit = recognize_digit(digit_slice, predefined_embeddings, model)
        detected_number.append(recognized_digit)

    # Join the recognized digits into a final number
    final_number = ''.join(map(str, detected_number))
    return final_number

if __name__ == "__main__":
    # Load ResNet50 model
    model = load_model()
    
    # Define paths
    input_image_path = 'extracted_frames/frame_15.png'
    predefined_digits_dir = 'predefined_digits'
    
    # Process the image and recognize digits
    final_number = process_image(input_image_path, predefined_digits_dir, model)
    print(f"The detected number is: {final_number}")
    cv2.destroyAllWindows()
