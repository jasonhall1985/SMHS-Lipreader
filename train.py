import argparse
import os
import sys
import time
import datetime
from model_utils import pretrain_lipnet_on_grid, initialize_lipnet_model

# Define the default path for GRID dataset relative to this script
# Assumes `train.py` is in the root of the project, and `data/` is a subdirectory.
DEFAULT_GRID_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "grid_dataset")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "lipnet_model.h5")

def check_grid_dataset(grid_path, required_speakers=5):
    """
    Check if the GRID dataset exists at the specified path and has the required structure.
    
    Args:
        grid_path: Path to the GRID dataset
        required_speakers: Minimum number of speakers required
        
    Returns:
        bool: True if the dataset is valid, False otherwise
    """
    if not os.path.exists(grid_path):
        print(f"ERROR: The specified GRID dataset path does not exist: {grid_path}")
        return False
        
    if not os.path.isdir(grid_path):
        print(f"ERROR: The specified GRID dataset path is not a directory: {grid_path}")
        return False
    
    # Check for speaker directories (s1, s2, etc.)
    speaker_count = 0
    total_videos = 0
    
    print("Checking dataset structure...")
    speaker_dirs = [d for d in os.listdir(grid_path) if os.path.isdir(os.path.join(grid_path, d)) and d.startswith('s')]
    
    # Sort speaker directories numerically
    speaker_dirs.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else float('inf'))
    
    for speaker_dir in speaker_dirs:
        speaker_path = os.path.join(grid_path, speaker_dir)
        
        # Check for video and alignment files
        videos = [f for f in os.listdir(speaker_path) if f.endswith('.mp4') or f.endswith('.mpg')]
        aligns = [f for f in os.listdir(speaker_path) if f.endswith('.align')]
        
        if videos and aligns:
            speaker_count += 1
            total_videos += len(videos)
            
        # Print progress for large datasets
        if speaker_count % 10 == 0 or speaker_count == len(speaker_dirs):
            print(f"  Checked {speaker_count}/{len(speaker_dirs)} speakers, found {total_videos} videos")
    
    if speaker_count == 0:
        print(f"ERROR: No valid speaker directories found with video (.mpg or .mp4) and .align files in {grid_path}")
        print("Please make sure you've downloaded and extracted the GRID corpus correctly.")
        return False
    
    if speaker_count < required_speakers:
        print(f"WARNING: Found only {speaker_count} valid speakers, but {required_speakers} were requested.")
        print(f"Training will proceed with the {speaker_count} available speakers.")
    
    print(f"Dataset verification complete: {speaker_count} speakers with {total_videos} videos")
    return True

def print_instructions():
    """Print instructions for downloading and setting up the GRID dataset."""
    print("\n===== GRID Dataset Instructions =====")
    print("The GRID corpus is needed for training the LipNet model.")
    print("1. Visit the official GRID corpus website to download:")
    print("   http://spandh.dcs.shef.ac.uk/gridcorpus/")
    print("2. Extract the downloaded files to the data/grid_dataset directory.")
    print("3. The directory structure should look like:")
    print("   data/grid_dataset/s1/  (Speaker 1)")
    print("   data/grid_dataset/s2/  (Speaker 2)")
    print("   ... and so on.")
    print("4. Each speaker directory should contain .mpg video files and corresponding .align files.")
    print("5. Make sure you have at least 5 speakers for optimal training.")
    print("==============================\n")

def print_separator():
    """Print a separator line."""
    print("-" * 50)

def format_time(seconds):
    """Format seconds into a human-readable time string."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def main():
    parser = argparse.ArgumentParser(description="LipNet Model Training Script for ICU Phrases")
    parser.add_argument("--grid_path", type=str, default=DEFAULT_GRID_DATA_PATH,
                        help=f"Path to the GRID dataset. Defaults to: {DEFAULT_GRID_DATA_PATH}")
    parser.add_argument("--speakers", type=int, default=100,
                        help="Number of speakers from GRID dataset to use for pretraining. Default is 100.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs. Default is 3.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size. Default is 8.")
    parser.add_argument("--force", action="store_true",
                        help="Force training even if the dataset check fails.")
    parser.add_argument("--check", action="store_true",
                        help="Only check if the dataset is valid, don't start training.")
    parser.add_argument("--steps_per_epoch", type=int, default=125,
                        help="Number of steps per epoch. Default is 125 (1000 videos / 8 batch size).")

    args = parser.parse_args()

    print_separator()
    print("LipNet Training for ICU Phrases")
    print_separator()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training start time: {timestamp}")
    print(f"GRID Dataset Path: {args.grid_path}")
    print(f"Number of Speakers: {args.speakers}")
    print(f"Training Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Steps per Epoch: {args.steps_per_epoch}")
    print_separator()
    
    # Check if the GRID dataset exists and has the required structure
    dataset_valid = check_grid_dataset(args.grid_path, args.speakers)
    
    if args.check:
        if dataset_valid:
            print("Dataset check PASSED. The GRID dataset is properly set up for training.")
        else:
            print("Dataset check FAILED. Please fix the issues above before training.")
            print_instructions()
        return
    
    if not dataset_valid and not args.force:
        print("Dataset check failed. Use --force to train anyway, or see instructions below:")
        print_instructions()
        return
    
    # Create model directories if they don't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        print(f"A trained model already exists at {MODEL_PATH}")
        user_input = input("Do you want to continue training this model? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Training aborted.")
            return
        print("Continuing training of existing model...")
    else:
        print("No existing model found. Will train from scratch.")
        # Initialize the model
        initialize_lipnet_model()
    
    print("Starting training process...")
    
    try:
        start_time = time.time()
        
        # Train the model
        pretrain_lipnet_on_grid(
            grid_data_path=args.grid_path, 
            num_speakers=args.speakers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch
        )
        
        training_time = time.time() - start_time
        
        print_separator()
        print(f"Training completed successfully in {format_time(training_time)}")
        print(f"Model saved to: {MODEL_PATH}")
        print("You can now run the Flask application to use this model:")
        print("  python app.py")
        print_separator()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("The partially trained model may have been saved.")
    except Exception as e:
        print(f"Error during training: {e}")
        print("Check the error message above and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 