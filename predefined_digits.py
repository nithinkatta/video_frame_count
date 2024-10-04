import cv2
import os

# Function to slice the input image into individual digits
def slice_image(image, num_slices, save_dir):
    height, width = image.shape
    slice_width = width // num_slices
    slices = []

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_slices):
        if i==4:
            digit_slice = image[:, i * slice_width+1:(i + 1) * slice_width+1]
        else:    
            digit_slice = image[:, i * slice_width:(i + 1) * slice_width]
        
        slices.append(digit_slice)
        
        # Save each slice as an image in the specified directory
        slice_path = os.path.join(save_dir, f'digit_slice_{i}.png')
        cv2.imwrite(slice_path, digit_slice)  # Save each slice as a PNG file
        print(f"Saved digit slice {i} at: {slice_path}")

    return slices

# Main function to process the image and save the slices
def save_sliced_images(input_image_path, save_dir):
    # Load input image
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Specify number of slices (adjust based on your image)
    num_slices = 5  # Adjust based on the number of digits in the image
    
    # Slice the input image and save the slices
    digit_slices = slice_image(input_image, num_slices, save_dir)
    
    return digit_slices

if __name__ == "__main__":
    # Define paths

    input_image_path = 'extracted_frames/frame_30.png'  # Path to the input image
    save_dir = 'predefined_digits'  # Directory to save the sliced images
    
    # Process the image and save the digit slices
    save_sliced_images(input_image_path, save_dir)
    print(f"All digit slices have been saved in the {save_dir} directory.")