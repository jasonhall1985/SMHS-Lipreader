# LipNet Flask Backend for ICU Phrases

A Flask-based backend application for lipreading using the LipNet model, specifically designed to recognize 10 predefined ICU phrases from video input.

## Features

- **Video Processing Pipeline**: Extracts and processes mouth regions from video frames
- **LipNet Model Integration**: Uses a deep learning model for lipreading
- **GRID Dataset Integration**: Pretrains on the GRID corpus dataset
- **REST API**: Simple `/predict` endpoint that accepts video uploads
- **ICU Phrase Prediction**: Maps predictions to 10 predefined ICU phrases

## Requirements

- Python 3.9+
- TensorFlow 2.13+
- OpenCV
- Flask
- GRID corpus dataset (for training)

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create required directories:
   ```
   mkdir -p uploads models data/grid_dataset
   ```

## ICU Phrases

The system recognizes the following 10 predefined ICU phrases:

1. "I'm in pain"
2. "I need suctioning"
3. "I can't breathe"
4. "I need water"
5. "I feel sick"
6. "I need repositioning"
7. "I need the toilet"
8. "I need a blanket"
9. "Call the nurse"
10. "I need my family"

## Training with GRID Dataset

The LipNet model needs to be trained before it can make accurate predictions. This implementation uses the GRID dataset for training.

### Getting the GRID Dataset

1. Download the GRID corpus from the [official website](http://spandh.dcs.shef.ac.uk/gridcorpus/)
2. Extract it to the `data/grid_dataset` directory
3. The structure should look like:
   ```
   data/grid_dataset/
   ├── s1/
   ├── s2/
   ├── s3/
   ├── s4/
   ├── s5/
   └── ...
   ```

### Training the Model

Run the following command to train the model:

```
python train.py --speakers 5 --epochs 10 --batch_size 4
```

Parameters:
- `--speakers`: Number of speakers to use from the GRID dataset (default: 5)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Training batch size (default: 4)
- `--grid_path`: Path to the GRID dataset (default: `data/grid_dataset`)
- `--force`: Force training even if dataset check fails
- `--check`: Only validate the dataset without training

The trained model will be saved to `models/lipnet_model.h5`.

## Running the Server

Start the Flask server with:

```
python app.py
```

or use the provided script:

```
./run.sh
```

The server will run at `http://127.0.0.1:5000`.

## API Usage

### Predict Endpoint

- **URL**: `/predict`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameter**: `file` - An .mp4 video file
- **Response**: JSON object with the predicted phrase and all available phrases
  ```json
  {
    "prediction": "I need water",
    "options": ["I'm in pain", "I need suctioning", ...]
  }
  ```

### Example Usage

Using `curl`:

```bash
curl -X POST -F "file=@example_video.mp4" http://127.0.0.1:5000/predict
```

Using the provided test script:

```bash
python test_api.py example_video.mp4
```

## Project Structure

```
.
├── app.py                # Flask application
├── model_utils.py        # LipNet model implementation and utilities
├── train.py              # Script for training on GRID dataset
├── test_api.py           # Script for testing the API
├── run.sh                # Shell script to start the server
├── requirements.txt      # Python dependencies
├── uploads/              # Directory for temporary storage of uploaded videos
├── models/               # Directory for storing trained models
├── data/                 # Data directory
│   ├── README.md         # Instructions for the GRID dataset
│   └── grid_dataset/     # GRID dataset (to be downloaded)
└── example_video.mp4     # Example video for testing
```

## Technical Implementation

- **Face/Mouth Detection**: Uses OpenCV Haar cascades
- **Video Preprocessing**: Extracts, normalizes, and sequences mouth regions
- **LipNet Architecture**: 3D CNNs + Bidirectional LSTMs with CTC loss
- **Training Pipeline**: Custom batch preparation optimized for the GRID dataset
- **Synthetic Labeling**: Maps GRID utterances to ICU phrases through rules and weighting

## Troubleshooting

- **No valid frames extracted**: Ensure the video shows a clear, well-lit face with visible mouth movements
- **GRID dataset not found**: Follow the instructions to download and extract the dataset properly
- **Training errors**: Check that your system has enough memory for the batch size specified

## Future Improvements

- Enhance mouth detection with more modern techniques (dlib, MediaPipe)
- Improve the mapping from GRID utterances to ICU phrases
- Add data augmentation for more robust training
- Create a web interface for easier testing

## Credits

- [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/) - Audiovisual corpus of speech
- [LipNet Paper](https://arxiv.org/abs/1611.01599) - Original LipNet research

## License

[MIT License]

## Contact

For questions or issues, please open an issue on this repository. 