import os
import random
import argparse
import numpy as np
from model_utils import ICU_PHRASES, initialize_lipnet_model, preprocess_video

def evaluate_phrase(model, phrase_dir, num_samples=5):
    """
    Evaluate the model on a specific ICU phrase
    
    Args:
        model: The trained LipNet model
        phrase_dir: Directory containing test videos for the phrase
        num_samples: Number of videos to test
        
    Returns:
        Accuracy and confidence scores for the phrase
    """
    # Get all the videos in the phrase directory
    videos = [os.path.join(phrase_dir, f) for f in os.listdir(phrase_dir) if f.endswith('.mp4')]
    
    # Determine which phrase this directory represents
    true_phrase = None
    for phrase in ICU_PHRASES:
        safe_phrase = phrase.lower().replace("'", "").replace(" ", "_")
        if safe_phrase in phrase_dir:
            true_phrase = phrase
            break
    
    if true_phrase is None:
        print(f"Error: Could not determine the true phrase for directory {phrase_dir}")
        return 0, []
    
    # Get true phrase index
    true_idx = ICU_PHRASES.index(true_phrase)
    
    # Select samples to test
    if len(videos) > num_samples:
        test_videos = random.sample(videos, num_samples)
    else:
        test_videos = videos
    
    correct = 0
    confidences = []
    
    for video_path in test_videos:
        # Preprocess the video
        processed_frames = preprocess_video(video_path)
        processed_frames = np.expand_dims(processed_frames, axis=0)
        
        # Run prediction
        predictions = model.predict(processed_frames, verbose=0)
        
        # Get the predicted phrase index and confidence
        pred_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_idx]
        
        # Get the predicted phrase
        pred_phrase = ICU_PHRASES[pred_idx]
        
        # Check if the prediction is correct
        is_correct = (pred_idx == true_idx)
        if is_correct:
            correct += 1
        
        # Store confidence score
        confidences.append(confidence)
        
        # Print individual result
        status = "✓" if is_correct else "✗"
        print(f"  {status} {os.path.basename(video_path)}: Predicted '{pred_phrase}' (Confidence: {confidence:.2f})")
    
    accuracy = correct / len(test_videos) if test_videos else 0
    return accuracy, confidences

def main():
    parser = argparse.ArgumentParser(description="Evaluate LipNet model on the 3 ICU phrases")
    parser.add_argument("--data_dir", type=str, default="data/training_videos",
                       help="Directory containing phrase subdirectories with test videos")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of videos to test for each phrase")
    
    args = parser.parse_args()
    
    # Load the trained model
    print("Loading LipNet model...")
    model = initialize_lipnet_model()
    
    # Results collection
    results = {}
    overall_accuracy = 0
    total_tested = 0
    
    # Evaluate each phrase
    for phrase in ICU_PHRASES:
        safe_phrase = phrase.lower().replace("'", "").replace(" ", "_")
        phrase_dir = os.path.join(args.data_dir, safe_phrase)
        
        if not os.path.exists(phrase_dir):
            print(f"Warning: Directory not found for phrase '{phrase}' at {phrase_dir}")
            continue
        
        print(f"\nEvaluating phrase: '{phrase}'")
        accuracy, confidences = evaluate_phrase(model, phrase_dir, args.samples)
        
        results[phrase] = {
            "accuracy": accuracy,
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "samples_tested": min(args.samples, len(os.listdir(phrase_dir)))
        }
        
        total_tested += results[phrase]["samples_tested"]
        overall_accuracy += accuracy * results[phrase]["samples_tested"]
    
    # Calculate overall accuracy (weighted by number of samples)
    if total_tested > 0:
        overall_accuracy /= total_tested
    
    # Print summary results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for phrase, data in results.items():
        print(f"'{phrase}':")
        print(f"  Accuracy: {data['accuracy']*100:.1f}%")
        print(f"  Average Confidence: {data['avg_confidence']:.2f}")
        print(f"  Samples Tested: {data['samples_tested']}")
        print()
    
    print(f"Overall Accuracy: {overall_accuracy*100:.1f}%")
    print("="*50)

if __name__ == "__main__":
    main() 