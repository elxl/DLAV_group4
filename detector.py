# Change directory
import os

path = os.getcwd()
os.chdir(path + "/yolov5-deepsort")

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, path + "/yolov5-deepsort")

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image

import torch
import torch.nn.functional as F

# import dependencies
from IPython.display import display, Javascript, Image
from base64 import b64decode, b64encode
import cv2
import PIL
import pandas as pd
from re import I

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from core.config import cfg
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

import mediapipe as mp

class Detector(object):
    """docstring for Detector"""
    def __init__(self):
        super(Detector, self).__init__()

        # INITIALISE YOLO DETECTION
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # pre-trained yolo model with the COCO dataset
        self.model.confidence = 0.4                                  # confidence threshold for person detection
        # read in all class names from config
        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # INITIALISE MEDIAPIPE POSE DETECTION
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # INITIALISE DEEPSORT :
        # Definition of the parameters for deepsort
        self.max_cosine_distance = 0.05   # threshold for the cosine distance between features. 
        self.nn_budget = 100              # number of previous frames of feature vectors should be retained for distance calculation for each track
        self.nms_max_overlap = 1.0        # Parameters in non maximum suppression

        self.model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        # calculate cosine distance metric
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        # initialize tracker
        self.tracker = Tracker(self.metric, max_age=40)

        # load configuration for object detector
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config=self.config)

        # Flags
        self.flaginfo = False # Print for debug

        # initialze bounding box to empty
        self.bbox = ''

        # Set all parameters for initialisation
        self.initialisation = True  # boolean to check if we are in the initialisation part (detection)
        self.count = 0              # counter to count the number of frame where we are detected
        self.IDofInterest = -1      # ID of our person of interest in deepsort. 
        self.empty_detectoin = 0    # counter to count the number of frame were where the person of interest is not tracked (used to do reinitialisation)
        self.tracking_interested = False # boolean to check if person of interest is found or not 

    def forward(self, frame):   
        
        frame_size = frame.shape[:2]
        frame = np.array(frame) # convert frame to numpy array
        
        # Detection
        result = self.model(frame) # inference using yolov5
        detectPerson = result.pandas().xyxy[0] # Get the bounding box, the confidence and the class
        detectPerson = detectPerson [(detectPerson['class']== 0)] # only select the person class 

        # Initialisze of the different variable 
        num_objects = 0         # number of person detected
        bboxes = np.array([])   # bounding box of the person detected
        scores = np.array([])   # confidence of the person detected
        classes = np.array([])  # class of the object detected (in our case always person)
        detectPersonOfInterest = pd.Series(dtype='float64') # bounding box for our person of interest
        
        # If we are in detection (finding the person of interest)
        if self.initialisation :
          print("initialisation")

          # crop the image on each person detected to apply pose detection on each person 
          for i in range(detectPerson.shape[0]):
            cur_person = detectPerson.iloc[i,:]
            xmin = int(cur_person['xmin'])
            xmax = int(cur_person['xmax'])
            ymin = int(cur_person['ymin'])
            ymax = int(cur_person['ymax'])
            frame_crop = np.ascontiguousarray(frame[ymin:(ymax+1),xmin:(xmax+1),:])

            frame_crop.flags.writeable = False 

            # Make pose detection
            results = self.pose.process(frame_crop)
            frame_crop.flags.writeable = True

            # extract shoulder and wrist values of pose detection
            if results.pose_landmarks:
              # Extract landmarks 
              landmarks = results.pose_landmarks.landmark
              left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
              right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
              left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
              right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

              # Check if the person have the correct pose (person of interest)
              if (left_wrist.y > left_shoulder.y) and (right_wrist.y < right_shoulder.y) and (left_wrist.x < left_shoulder.x) and (left_wrist.x > right_shoulder.x) and (right_wrist.x < left_shoulder.x) and (right_wrist.x > right_shoulder.x) :
                # Write everything in the format to give it to deepsort model : 
                detectPersonOfInterest = detectPerson.iloc[i,:]          
                num_objects=1
                bboxes = np.zeros((1,4))
                scores = np.zeros(1)
                classes = np.zeros(1)
                bboxes[0,0] = int(detectPersonOfInterest['ymin'])/frame_size[1]
                bboxes[0,1] = int(detectPersonOfInterest['xmin'])/frame_size[0]
                bboxes[0,2] = int(detectPersonOfInterest['ymax'])/frame_size[1]
                bboxes[0,3] = int(detectPersonOfInterest['xmax'])/frame_size[0]
                scores[0] = np.array(float(detectPerson.iloc[0]['confidence']))
                classes[0] = np.array(0) # class person

                # Do this detection step for 10 frames so that we are more robust on the detection 
                self.count += 1 
                if self.count > 10: 
                  print("***************initialisation finish******************")
                  self.initialisation = False
                break                     

        # If not in initialisation send all the person detected to the deepsort model so that it can track them 
        elif not detectPerson.empty:
        
          # Put all person detected into deepsort format 
          num_objects = detectPerson.shape[0]

          bboxes = np.zeros((num_objects,4))
          scores = np.zeros(num_objects)
          classes = np.zeros(num_objects)

          for i in range(num_objects):
            bboxes[i,0] = int(detectPerson.iloc[i]['ymin'])/frame_size[1]
            bboxes[i,1] = int(detectPerson.iloc[i]['xmin'])/frame_size[0]
            bboxes[i,2] = int(detectPerson.iloc[i]['ymax'])/frame_size[1]
            bboxes[i,3] = int(detectPerson.iloc[i]['xmax'])/frame_size[0]
            scores[i] = np.array(float(detectPerson.iloc[i]['confidence']))
            classes[i] = np.array(0) # class person

        else:
          # If no one detected give empty array to deepsort so there is no tracking 
          num_objects = 0
          bboxes = np.array([])
          scores = np.array([])
          classes = np.array([])

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w = frame.shape[0], frame.shape[1]
        bboxes = utils.format_boxes(bboxes, original_h, original_w)


        # loop through objects and use class index to get class name
        names = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = self.class_names[class_indx]
            names.append(class_name)


        # encode yolo detections and feed to tracker
        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       
        
        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)
        self.tracking_interested = False

        # center of bounding box and label that we will give to the lomoo robot : 
        bbox_interested = [0,0]
        class_name_interested = None

        # update tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # If Initialisation, take the ID of the person of interest if founded 
            if self.initialisation and not detectPersonOfInterest.empty:
              self.IDofInterest = track.track_id

            # Track the person of interest
            if track.track_id == self.IDofInterest :
              self.tracking_interested = True
              self.empty_detectoin = 0
              bbox_interested[1] = (bbox[1] + bbox[3])/2
              bbox_interested[0] = (bbox[0] + bbox[2])/2
              class_name_interested = [1.0]
        
        # Reidentification in case we lost track of the person of interest
        if not self.initialisation and not self.tracking_interested:
          self.empty_detectoin += 1 
        
        if self.empty_detectoin > 40:
          print("--------------redo initialization---------------")
          self.initialisation = True  
          self.IDofInterest = -1
          self.empty_detectoin = 0
          self.count = 0

        # If the person of interest is not detected or tracked : give those values so that the robot doesn't move
        if bbox_interested[0] == 0 and bbox_interested[1] == 0:
          return [80,60], [0.0] 
        
        return bbox_interested, class_name_interested

# Test the detector on our webcam 
detector = Detector()
from PIL import Image
cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    #get the frame
    ret, frame = cap.read()
    frame = np.array(frame)
    frame = cv2.resize(frame,(int(160),int(120)), interpolation=cv2.INTER_AREA)

    # run detector
    pred_box,label = detector.forward(frame)
    
    # show results 
    cv2.rectangle(frame, (int(pred_box[0]-5), int(pred_box[1]-5)), (int(pred_box[0]+5), int(pred_box[1]+5)), [250,0,0], 2)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 27:
       break

cap.release()
cv2.destroyAllWindows()
"""
if __name__ == "__main__":
  detector = Detector()
  frame = cv2.imread("../0001.jpg")
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  bbox, bbox_label = detector.forward(frame)
  print(bbox)
"""