# CS-459 EPFL Deep learning for autonomous vehicles: Final Project 
Group 4: Nada Guerraoui, Thomas Peeters, Xinling Li, Kuan Lon Vu

This is the repository for the final project of CS-459. The goal of this project is to train a detector and tracker which will be exported to run on a Loomo robot at the EPFL Tandem Race 2022. For the model, we decided to use YOLOv5 deepsort as our selected model for the race.

## Repository Structure
This repository contains:
```
DLAV_group4
│   0001.jpg
│   README.md
│   client.py
│   detect_tracking.ipynb
│   detector.py
│   main.py
│   requirements.txt
│   run_client.sh
│   saved_model.pth
│   setup.py
│   test.py
│
└───yolov5-deepsort 
    
```
### YOLOv5 + deepsort files
`yolov5-deepsort`: This folder contains all the code for YOLOv4 for detection (since DeepSORT runs detection at every frame) and DeepSORT for tracking, cloned from [theAIGuysCode](https://github.com/theAIGuysCode/yolov4-deepsort). 

### Our model 
`detector_tracking.ipynb`: Our tracker implemented in `.ipynb` file. 
`client.py`: This file communicates between `detector.py` and the V100's server.
`detector.py`: Our tracker implemented in this file to be compatible with `client.py`
`main.py`: `detect_tracking.ipynb` into a `.py` file.

## Requirements
To install the required packages, run:

`pip install -r requirements.txt`

## Instructions to run the code
*Note: Please connect to the EPFL network before proceeding*

### Connect to the V100 server
Run:

`ssh -AY group4@128.178.240.162` (Enter password when prompted)

`source venv/bin/activate`

### Connect to the Loomo
*Note: Please turn on the Loomo, find its dynamic IP address and start the robot app on it before proceeding*
Run:

`adb connect <Loomo IP address>`

### Run tracker on the Loomo
Run:

`python client.py --ip-address <Loomo IP address>`

### Initialise the tracker
Stand in front of the camera, do the following pose. Please hold the pose for 10 seconds.

TODO: Insert a photo of the initalisation gesture


### References
- theAIGuysCode (2022). yolov4-deepsort. https://github.com/theAIGuysCode/yolov4-deepsort