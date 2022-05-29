# Deep learning for autonomous vehicles Final Project 
Group 4: Nada Guerraoui, Thomas Peeters, Xinling Li, Kuan Lon Vu

## Repository Structure
TODO:
```
```

## Milestone 1: Person detection
### Instructions to run the code
TODO:
To run the code for milestone 1, it is required to set it up on Google Colab as the code involves using the JS code scippets to access the webcam. User can upload the files to Google Colab (or to Google Drive and then mount the Drive to Google Colab).

Once the code is set up on Google Colab, please click run 


## Milestone 2
### Instructions to run the code
TODO:
Upload the YOLO weights to `` folder ....

## Milestone 3
### Instructions to run the code
*Note: Please connect to the EPFL network before proceeding*

#### Connect to the V100 server
Run:

`ssh -AY group4@128.178.240.162`

`source venv/bin/activate`

#### Connect to the Loomo
*Note: Please turn on the Loomo, find its dynamic IP address and start the robot app on it before proceeding*
Run:

`adb connect <Loomo IP address>`

#### Run tracker on the Loomo
Run:

`python client.py --ip-address <Loomo IP address>`

#### Initialise the tracker
Stand in front of the camera, do the following pose. Please hold the pose for 10 seconds.

TODO: Insert a photo of the initalisation gesture
