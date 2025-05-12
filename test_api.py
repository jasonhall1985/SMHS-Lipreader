#!/usr/bin/env python
import os
import requests
import json
import argparse
import sys

def test_predict_endpoint(video_path, url="http://127.0.0.1:5000/predict"):
    """
    Test the /predict endpoint of the LipNet Flask API.
    
    Args:
        video_path: Path to an .mp4 video file for testing
        url: URL of the API endpoint
    
    Returns:
        0 if successful, non-zero otherwise
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return 1
        
    if not video_path.lower().endswith('.mp4'):
        print(f"Error: File must be an .mp4 video, got {video_path}")
        return 1
    
    print(f"Testing /predict endpoint with video: {video_path}")
    print(f"API URL: {url}")
    print("Sending request...")
    
    try:
        with open(video_path, 'rb') as video_file:
            files = {'file': (os.path.basename(video_path), video_file, 'video/mp4')}
            response = requests.post(url, files=files)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nAPI Response:")
            print(f"Predicted phrase: \"{result['prediction']}\"")
            print("\nAll available phrases:")
            for i, phrase in enumerate(result['options'], 1):
                print(f"  {i}. {phrase}")
            print("\nTest completed successfully!")
            return 0
        else:
            print(f"Error: API returned status code {response.status_code}")
            print("Response content:")
            print(response.text)
            return 1
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the Flask server is running at http://127.0.0.1:5000")
        print("You can start it with: python app.py")
        return 1
        
    except Exception as e:
        print(f"Error during API test: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Test the LipNet API /predict endpoint")
    parser.add_argument("video_path", type=str, help="Path to an .mp4 video file to test with")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:5000/predict", 
                       help="URL of the API endpoint (default: http://127.0.0.1:5000/predict)")
    
    args = parser.parse_args()
    
    return test_predict_endpoint(args.video_path, args.url)

if __name__ == "__main__":
    sys.exit(main()) 