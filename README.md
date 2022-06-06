# CS-459 EPFL Deep learning for autonomous vehicles: Final Project 
Group 4: Xinling Li, Nada Guerraoui, Thomas Peeters, Kuan Lon Vu

This is the repository for the final project of CS-459. The goal of this project is to train a detector and tracker which will be exported to run on a Loomo robot at the EPFL Tandem Race 2022. For the model, we decided to use YOLOv5 deepsort as our selected model for the race.

## Repository Structure
This repository contains:
```
DLAV_group4
│   client.py
│   requirements.txt
│   initialisation_pose.jpg
│   Milestone2.ipynb
│   README.md
│   detector.py
│
└───yolov5-deepsort
    
```
### YOLOv5 + deepsort files
*These files are used in milestones 2 and 3.*

`yolov5-deepsort`: This folder contains all the code for YOLOv5 for detection (since DeepSORT runs detection at every frame) and DeepSORT for tracking, cloned from [theAIGuysCode](https://github.com/theAIGuysCode/yolov4-deepsort). We have added the YOLOv5 weights (`yolov5s.pt`) in the folder.

### Milestone 1: detection

`Milestone2.ipynb`: Our YOLOv5 deepsort tracker which needs to run on Google Colab.
`best_mask_weight` : The trained weights for mask detection

### Milestone 2: tracking
`Milestone2.ipynb`: Our YOLOv5 deepsort tracker which needs to run on Google Colab.

### Milestone 3: Tandom Race

`client.py`: This file communicates between `detector.py` and the V100's server.

`detector.py`: Our tracker implemented in this file to be compatible with `client.py`.


## Instructions to run the code
*Note: Please connect to the EPFL network before proceeding*

### Milestones 1 and 2
To run the code for milestones 1 and 2, it is required to set it up on Google Colab as the code involves using the JS code scippets to access the webcam. Please upload the files to Google Colab (or to Google Drive and then mount the Drive to Google Colab). A full tutorial to set up Google Colab is beyond the scope of this documentation.

Please ensure that the machine running the code is connect to the webcam.

### Milestone 3
Milestone 3 builds on the code in Milestone 2, but for the detection, we have opted to use pose detection using Mediapipe.

#### Requirements
To install the required packages, run:

`pip install -r requirements.txt`

#### Connect to the V100 server
Run:

`ssh -AY group4@128.178.240.162` (Enter password when prompted)

`source venv/bin/activate`

#### Connect to the Loomo
*Note: Please turn on the Loomo, find its dynamic IP address and start the robot app on it before proceeding*
Run:

`adb connect <Loomo IP address>`

#### Run tracker on the Loomo
Run:

`python client.py --ip-address <Loomo IP address>`

### Initialise the tracker
Stand in front of the camera, do the following pose. Please hold the pose until initialisation is complete.

![Initialisation pose](initialisation_pose.jpg)


## References
- theAIGuysCode (2022). yolov4-deepsort. https://github.com/theAIGuysCode/yolov4-deepsort
- Camillo Lugaresi et al. “MediaPipe: A Framework for Building Perception Pipelines”. In: arXiv e-prints, arXiv:1906.08172 (June 2019), arXiv:1906.08172. arXiv: [1906.08172](https://arxiv.org/abs/1906.08172) [cs.DC].
