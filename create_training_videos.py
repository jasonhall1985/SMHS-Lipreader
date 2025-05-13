import os
import numpy as np
import cv2
import random
from model_utils import ICU_PHRASES

# Ensure directories exist
os.makedirs('data/training_videos', exist_ok=True)
os.makedirs('data/training_videos/im_in_pain', exist_ok=True)
os.makedirs('data/training_videos/need_suctioning', exist_ok=True)
os.makedirs('data/training_videos/cant_breathe', exist_ok=True)

# Number of videos to generate per phrase
NUM_VIDEOS_PER_PHRASE = 20
TOTAL_VIDEOS = NUM_VIDEOS_PER_PHRASE * len(ICU_PHRASES)

# Video parameters
width, height = 640, 480
fps = 25
duration = 3  # seconds
num_frames = int(duration * fps)

# Phrase-specific mouth movement patterns
phrase_patterns = {
    "I'm in pain": {
        "frequency": 0.35,  # Medium frequency for "I'm in pain"
        "amplitude": 18,    # Medium opening
        "color": (0, 30, 180),  # Red mouth
        "expression": "pain" 
    },
    "I need suctioning": {
        "frequency": 0.5,   # High frequency for "I need suctioning"
        "amplitude": 15,    # Small to medium opening
        "color": (30, 30, 140),  # Pink mouth
        "expression": "round"
    },
    "I can't breathe": {
        "frequency": 0.25,  # Low frequency for "I can't breathe"
        "amplitude": 25,    # Larger opening
        "color": (20, 20, 170),  # Red-blue mouth
        "expression": "gasping"
    }
}

# Function to add noise/variation to make videos more realistic
def add_variation(base_value, variation_percent=0.2):
    variation = base_value * variation_percent
    return base_value + random.uniform(-variation, variation)

def create_video_for_phrase(phrase, video_index):
    pattern = phrase_patterns[phrase]
    
    # Create filename and path based on phrase type
    safe_phrase = phrase.lower().replace("'", "").replace(" ", "_")
    output_file = f"data/training_videos/{safe_phrase}/video_{video_index:03d}.mp4"
    
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Add some randomness to face position for variation
    face_x_offset = random.randint(-40, 40)
    face_y_offset = random.randint(-30, 30)
    face_size_var = random.uniform(0.9, 1.1)
    
    # Add lighting/color variation
    brightness = random.uniform(0.8, 1.2)
    
    # Generate frames with a simulated face and mouth movement
    for i in range(num_frames):
        # Create a background with some variation
        bg_color = np.random.randint(180, 220)
        frame = np.ones((height, width, 3), dtype=np.uint8) * bg_color
        
        # Draw a face oval
        face_center = (width // 2 + face_x_offset, height // 2 + face_y_offset)
        face_axes = (int(150 * face_size_var), int(200 * face_size_var))
        face_color = (200, 200, 200)  # Light gray base face color
        cv2.ellipse(frame, face_center, face_axes, 0, 0, 360, face_color, -1)
        
        # Add some contour to the face
        cv2.ellipse(frame, face_center, face_axes, 0, 0, 360, (180, 180, 180), 2)
        
        # Draw eyes
        eye_y_pos = face_center[1] - int(50 * face_size_var)
        eye_x_offset = int(65 * face_size_var)
        eye_radius = int(25 * face_size_var)
        
        left_eye_center = (face_center[0] - eye_x_offset, eye_y_pos)
        right_eye_center = (face_center[0] + eye_x_offset, eye_y_pos)
        
        # White of the eyes
        cv2.circle(frame, left_eye_center, eye_radius, (255, 255, 255), -1)
        cv2.circle(frame, right_eye_center, eye_radius, (255, 255, 255), -1)
        
        # Pupils - make them blink occasionally
        pupil_radius = int(10 * face_size_var)
        blink = (i % 50 > 45) or (random.random() > 0.95)  # Blink timing
        
        if not blink:
            # Position of pupil varies slightly to simulate eye movement
            left_pupil_x = left_eye_center[0] + random.randint(-5, 5)
            left_pupil_y = left_eye_center[1] + random.randint(-3, 3)
            right_pupil_x = right_eye_center[0] + random.randint(-5, 5)
            right_pupil_y = right_eye_center[1] + random.randint(-3, 3)
            
            cv2.circle(frame, (left_pupil_x, left_pupil_y), pupil_radius, (0, 0, 0), -1)
            cv2.circle(frame, (right_pupil_x, right_pupil_y), pupil_radius, (0, 0, 0), -1)
        else:
            # Draw closed eyelids
            cv2.ellipse(frame, left_eye_center, (eye_radius, 2), 0, 0, 180, (120, 120, 120), 2)
            cv2.ellipse(frame, right_eye_center, (eye_radius, 2), 0, 0, 180, (120, 120, 120), 2)
        
        # Draw a mouth that moves according to the phrase pattern
        mouth_x = face_center[0] + random.randint(-10, 10)  # Small random position variation
        mouth_y = face_center[1] + int(60 * face_size_var)
        
        # Calculate the height/opening of the mouth based on the phrase pattern
        frequency = add_variation(pattern["frequency"], 0.1)
        amplitude = add_variation(pattern["amplitude"], 0.2)
        
        # Different mouth movement patterns
        if pattern["expression"] == "pain":
            # Pain expression - more erratic
            mouth_height = int(amplitude * (0.5 + 0.5 * abs(np.sin(i * frequency + random.random() * 0.2))))
            # Make corners of mouth down slightly for pain expression
            mouth_width = int(50 * face_size_var)
            
            # Draw the mouth with slight frown
            pts = np.array([
                [mouth_x - mouth_width, mouth_y],
                [mouth_x - mouth_width//2, mouth_y + mouth_height//2],
                [mouth_x, mouth_y + mouth_height//4],
                [mouth_x + mouth_width//2, mouth_y + mouth_height//2],
                [mouth_x + mouth_width, mouth_y]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [pts], pattern["color"])
            
        elif pattern["expression"] == "gasping":
            # Gasping/can't breathe - wide open, then closing
            mouth_height = int(amplitude * (0.3 + 0.7 * abs(np.sin(i * frequency))))
            mouth_width = int(45 * face_size_var) + int(10 * abs(np.sin(i * frequency)))
            
            # Draw more oval mouth for gasping
            cv2.ellipse(frame, (mouth_x, mouth_y), (mouth_width, mouth_height), 0, 0, 360, pattern["color"], -1)
            
            # Add darkness inside mouth for depth
            cv2.ellipse(frame, (mouth_x, mouth_y + mouth_height//3), 
                        (mouth_width-10, max(1, mouth_height-10)), 0, 0, 360, (20, 20, 40), -1)
            
        else:  # Round/normal mouth for "I need suctioning"
            # Standard rounded mouth
            mouth_height = int(amplitude * (0.2 + 0.8 * abs(np.sin(i * frequency))))
            mouth_width = int(40 * face_size_var)
            
            # Draw the mouth as an ellipse
            cv2.ellipse(frame, (mouth_x, mouth_y), (mouth_width, mouth_height), 0, 0, 360, pattern["color"], -1)
        
        # Add text indicating which ICU phrase this simulates
        cv2.putText(frame, f"Phrase: '{phrase}'", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Write the frame to the video
        video.write(frame)
    
    # Release the VideoWriter
    video.release()
    
    return output_file

print(f"Generating {TOTAL_VIDEOS} training videos...")

# Generate videos for each phrase
videos_created = {}
for phrase in ICU_PHRASES:
    videos_created[phrase] = []
    for i in range(NUM_VIDEOS_PER_PHRASE):
        video_file = create_video_for_phrase(phrase, i+1)
        videos_created[phrase].append(video_file)
        
        # Print progress
        video_count = sum(len(v) for v in videos_created.values())
        print(f"Progress: {video_count}/{TOTAL_VIDEOS} videos created ({video_count/TOTAL_VIDEOS*100:.1f}%)",
              f"- Created: {video_file}")

print("\nSynthetic training data generation complete!")
print(f"Total videos: {TOTAL_VIDEOS}")

# Print counts by category
for phrase in ICU_PHRASES:
    safe_phrase = phrase.lower().replace("'", "").replace(" ", "_")
    count = len(videos_created[phrase])
    print(f"  - '{phrase}': {count} videos in 'data/training_videos/{safe_phrase}/'")

print("\nYou can now train the model using:")
print("python train.py --grid_path data/training_videos --speakers 3 --epochs 15 --batch_size 4 --steps_per_epoch 15") 