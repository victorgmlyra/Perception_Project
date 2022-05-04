# Perception_Project

Project for the course **31392 Perception for autonomous systems**

## Installation

1. Set up a conda environment:

    ```bash
    conda create -n <name> python=3.8
    conda activate <name>
    ```

2. Install pytorch and torchvision:

    ```bash
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    ```

3. Install the rest of the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data

- To run without training, download the pre-trained model [here](https://drive.google.com/file/d/15pqvgD2hQbPn16eabIEY3EN6aEiRiSmT/view?usp=sharing) and add it inside the models folder. 
- To train a new model, download the training data [here](https://drive.google.com/file/d/1D0SWy6GJX9jDJNu1pgumCChnPK7YuG4D/view?usp=sharing) and add it inside the dataset folder.
- Download the [calibration data](https://dtudk-my.sharepoint.com/personal/evanb_dtu_dk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fevanb%5Fdtu%5Fdk%2FDocuments%2FCourses%2F31392%2FFinal%5FProject%2FStereo%5Fcalibration%5Fimages%2Erar&parent=%2Fpersonal%2Fevanb%5Fdtu%5Fdk%2FDocuments%2FCourses%2F31392%2FFinal%5FProject&ga=1), [data w/o occlusion](https://dtudk-my.sharepoint.com/personal/evanb_dtu_dk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fevanb%5Fdtu%5Fdk%2FDocuments%2FCourses%2F31392%2FFinal%5FProject%2FStereo%5Fconveyor%5Fwithout%5Focclusions%2Erar&parent=%2Fpersonal%2Fevanb%5Fdtu%5Fdk%2FDocuments%2FCourses%2F31392%2FFinal%5FProject&ga=1), and [data with occlusion](https://dtudk-my.sharepoint.com/personal/evanb_dtu_dk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fevanb%5Fdtu%5Fdk%2FDocuments%2FCourses%2F31392%2FFinal%5FProject%2FStereo%5Fconveyor%5Fwith%5Focclusions%2Erar&parent=%2Fpersonal%2Fevanb%5Fdtu%5Fdk%2FDocuments%2FCourses%2F31392%2FFinal%5FProject&ga=1).
- The data folder should look like this:
    ```
    data/
    │
    └─── dataset/
    |    │
    |    └─ Training dataset
    │
    └─── models/
    |    │
    |    └─ Pre-trained model
    │
    └─── Stereo_calibration_images/*
    │
    └─── Stereo_conveyor_without_occlusions/*
    │
    └─── Stereo_conveyor_with_occlusions/*
    │
    └─── calibration.pkl
    ```

### Running

While inside the conda environment created:

- Run ```calibrate.py``` to calibrate the cameras
- Run ```train.py``` to train a model
- Run ```test.py``` to test the model
- Run ```run.py``` to run the full project
