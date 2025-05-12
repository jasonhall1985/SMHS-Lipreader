import os
import numpy as np
import cv2

# Create a test video file for the /predict endpoint
output_file = 'example_video.mp4'

# Video parameters
width, height = 640, 480
fps = 25
duration = 3  # seconds
num_frames = int(duration * fps)

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Generate frames with a simulated face and mouth movement
for i in range(num_frames):
    # Create a black frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a face oval
    face_center = (width // 2, height // 2)
    face_axes = (150, 200)
    cv2.ellipse(frame, face_center, face_axes, 0, 0, 360, (100, 100, 100), -1)
    
    # Draw eyes
    eye_radius = 25
    left_eye_center = (face_center[0] - 65, face_center[1] - 50)
    right_eye_center = (face_center[0] + 65, face_center[1] - 50)
    cv2.circle(frame, left_eye_center, eye_radius, (255, 255, 255), -1)
    cv2.circle(frame, right_eye_center, eye_radius, (255, 255, 255), -1)
    
    # Draw pupils
    pupil_radius = 10
    cv2.circle(frame, left_eye_center, pupil_radius, (0, 0, 0), -1)
    cv2.circle(frame, right_eye_center, pupil_radius, (0, 0, 0), -1)
    
    # Draw a mouth that opens and closes to simulate speaking
    # Calculate the height of the mouth based on frame index
    # This creates a sinusoidal motion
    mouth_height = int(10 + 20 * abs(np.sin(i * 0.5)))
    mouth_x = face_center[0]
    mouth_y = face_center[1] + 60
    
    # Draw the mouth as an ellipse
    mouth_color = (0, 0, 200)  # Red
    cv2.ellipse(frame, (mouth_x, mouth_y), (50, mouth_height), 0, 0, 360, mouth_color, -1)
    
    # Add text indicating which ICU phrase this simulates
    cv2.putText(frame, "Test: 'I need water'", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Write the frame to the video
    video.write(frame)

# Release the VideoWriter
video.release()

print(f"Test video created: {output_file}")
print("You can now test the API with this video:")
print(f"  python test_api.py {output_file}")