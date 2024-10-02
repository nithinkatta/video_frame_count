import cv2
import os

# Define the video file path
video_path = 'video.mp4'

# Create a directory to store extracted frames
frames_dir = 'extracted_frames'
os.makedirs(frames_dir, exist_ok=True)

# Initialize the video capture object
cap = cv2.VideoCapture(video_path)

# Initialize a frame counter
frame_count = 0
saved_frames = []

# Read through the video frame by frame
def crop_frame(frame, region):
    x, y, w, h = region
    return frame[y:y+h, x:x+w]  # Crop the frame based on region


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    region_of_interest = (113, 115, 150, 33)  # Example values, adjust as needed
    frame = crop_frame(frame, region_of_interest)
    # Save every frame to the directory (for now, saving all)
    frame_filename = os.path.join(frames_dir, f'frame_{frame_count}.png')
    cv2.imwrite(frame_filename, frame)
    saved_frames.append(frame_filename)
    
    frame_count += 1

cap.release()

frame_count, saved_frames[:5]  # Return total frames and first 5 frame file names for reference
