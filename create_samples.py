import os
import numpy as np
import cv2

# Create directory for GRID dataset if it doesn't exist
os.makedirs('data/grid_dataset', exist_ok=True)

# Parameters for sample videos
num_speakers = 100  # Increased to 100 speakers
num_samples_per_speaker = 10  # 10 videos per speaker = 1000 total
video_length = 20  # Reduced for faster generation
width, height = 100, 100

# Enhanced sample text for alignment files - more variety
sample_texts = [
    "bin blue at f two now",
    "place red by s five soon",
    "set green in c six please",
    "lay white with p one again",
    "bin red on j three soon",
    "put green at u nine please",
    "set white in e six now",
    "lay blue with r eight soon",
    "bin green in x seven again",
    "place white by t three now",
    "set red by j four please",
    "lay blue with u six again",
    "put white in h nine please",
    "bin green at m two soon",
    "place red with n five now",
    "set blue by k seven please",
    "lay green with v four again",
    "bin white in z one soon",
    "place blue at y three please",
    "put red by q eight now"
]

# Record which speakers we're generating
existing_speakers = range(1, 16)  # s1-s15 already exist
new_speakers = range(16, 101)     # s16-s100 are new

print(f"Starting generation of videos from {len(new_speakers)} new speakers...")

# Generate sample videos for each speaker
for speaker_idx in new_speakers:  # Only generate for new speakers
    # Create speaker directory
    speaker_dir = f'data/grid_dataset/s{speaker_idx}'
    os.makedirs(speaker_dir, exist_ok=True)
    
    for sample_idx in range(1, num_samples_per_speaker + 1):
        # Create a simple video
        video_path = f'{speaker_dir}/sample_{sample_idx}.mp4'
        
        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 25, (width, height))
        
        # Generate frames (simple moving gradient)
        for frame_idx in range(video_length):
            # Create a simple gradient image that changes over time
            gradient = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(width):
                color_value = int(255 * (i + frame_idx) / (width + video_length))
                gradient[:, i, :] = [color_value, color_value, color_value]
            
            # Add a simple circle representing a mouth
            # Make movement patterns slightly different for each speaker
            center_x = width // 2 + int(5 * np.sin(speaker_idx * 0.1 + frame_idx * 0.1))
            center_y = height // 2 + int(10 * np.sin(frame_idx * 0.3 + speaker_idx * 0.05))
            
            # Different color for different speakers
            r = (speaker_idx * 13) % 255
            g = (speaker_idx * 17) % 255
            b = (speaker_idx * 19) % 255
            
            # Different sizes for different speakers
            radius = 10 + (speaker_idx % 10)
            
            cv2.circle(gradient, (center_x, center_y), radius, (b, g, r), -1)
            
            # Write the frame
            video.write(gradient)
        
        # Release the video
        video.release()
        
        # Create corresponding alignment file
        align_path = video_path.replace('.mp4', '.align')
        with open(align_path, 'w') as f:
            # Select a sample text with some variation
            text_index = (sample_idx + speaker_idx) % len(sample_texts)
            text = sample_texts[text_index]
            f.write(text)
        
        # Print progress every 100 videos
        videos_created = (speaker_idx - new_speakers[0]) * num_samples_per_speaker + sample_idx
        total_to_create = len(new_speakers) * num_samples_per_speaker
        if videos_created % 100 == 0 or videos_created == total_to_create:
            print(f"Progress: {videos_created}/{total_to_create} videos created ({videos_created/total_to_create*100:.1f}%)")

# Count total videos
existing_videos = len(existing_speakers) * num_samples_per_speaker
new_videos = len(new_speakers) * num_samples_per_speaker
total_videos = existing_videos + new_videos

print(f"Sample GRID dataset complete with {total_videos} videos from {num_speakers} speakers")
print(f"Added {new_videos} new videos from speakers s{new_speakers[0]}-s{new_speakers[-1]}")
print("The dataset is now ready for training") 