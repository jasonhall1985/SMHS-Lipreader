# GRID Dataset Instructions

This directory is intended to house the GRID dataset, which is used for pretraining the LipNet model as outlined in `model_utils.py` and `train.py`.

## 1. Downloading the GRID Dataset

The GRID dataset is a large audiovisual sentence corpus.

- **Official Source**: You can typically find information and download links for the GRID dataset from the University of Sheffield's website or related academic sources. Search for "GRID corpus dataset".
- **Structure**: The dataset is usually structured into directories for each speaker (e.g., `s1`, `s2`, ..., `s34`). Each speaker directory contains video files (often `.mpg`) and corresponding alignment files (`.align`) which describe the timing of words spoken.

## 2. Placement

- Create a subdirectory named `grid_dataset` (or similar, and update the path in `train.py` if you choose a different name) inside this `data/` directory.
- Extract/place the GRID dataset speaker folders (e.g., `s1`, `s2`, etc.) directly into your `data/grid_dataset/` directory.

Your directory structure should look something like this:

```
<your_project_root>/
|-- app.py
|-- model_utils.py
|-- train.py
|-- requirements.txt
|-- uploads/
|   |-- .gitkeep
|-- data/
|   |-- README.md
|   |-- grid_dataset/      <-- Place GRID dataset here
|   |   |-- s1/
|   |   |   |-- bbaf2n.mpg
|   |   |   |-- bbaf2n.align
|   |   |   |-- ... (other videos and alignments for speaker 1)
|   |   |-- s2/
|   |   |   |-- ... (files for speaker 2)
|   |   |-- ... (other speaker folders up to s34)
|-- example_video.mp4
```

## 3. Usage in Pretraining

The `pretrain_lipnet_on_grid` function in `model_utils.py` (called by `train.py`) expects the path to this `grid_dataset` directory.

- The function will attempt to load video and alignment data for a specified number of speakers (default is 5, e.g., `s1` through `s5`).
- You will need to implement the actual data loading, preprocessing, synthetic labeling (mapping GRID utterances to the 10 ICU phrases), and LipNet model training logic within the placeholder sections of `model_utils.py`.

## Synthetic Labeling Note

Mapping GRID utterances (e.g., "bin blue at f 2 now") to the 10 predefined ICU phrases (e.g., "I'm in pain") is a non-trivial task and is critical for the model to learn relevant features. The current `pretrain_lipnet_on_grid` function contains a placeholder for this step. You'll need to devise a strategy for this, such as:
- Keyword-based mapping.
- Using a subset of GRID utterances that can be more easily mapped.
- Generating or augmenting data to fit the ICU phrases.

This setup provides the framework; the core ML work of data processing and model training for LipNet using GRID requires significant implementation. 