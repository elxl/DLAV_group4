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
import matplotlib.pyplot as plt
import os
from PIL import Image

import torch
import torch.nn.functional as F

# import dependencies
from IPython.display import display, Javascript, Image
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
from PIL import Image
import io
import html
import time
import matplotlib.pyplot as plt
import pandas as pd

from re import I
import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from absl.flags import FLAGS
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic

        # Definition of the parameters
        self.max_cosine_distance = 0.05
        self.nn_budget = 100
        self.nms_max_overlap = 1.0
        
        # initialize deep sort
        self.model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        # calculate cosine distance metric
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        # initialize tracker
        self.tracker = Tracker(self.metric, max_age=200)

        # load configuration for object detector
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config=self.config)


        # Flags
        self.flaginfo = False


        # read in all class names from config
        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # initialze bounding box to empty
        self.bbox = ''
        self.frame_num = 0
        self.initialisation = True
        self.count = 0
        self.IDofInterest = -1

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.confidence = 0.4

        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def load(self, PATH):
        # self.model = torch.load(PATH)
        # self.model.eval()

        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()

    def forward(self, frame):   
        # grayscale image for face detection
        # frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        frame_size = frame.shape[:2]

        # create transparent overlay for bounding box
        bbox_array = np.zeros([frame_size[0],frame_size[1],4], dtype=np.uint8) 
        

        # Detection
        result = self.model(frame) # inference for person
        detectPerson = result.pandas().xyxy[0] # Get the bounding box, the confidence and the class
        detectPerson = detectPerson [(detectPerson['class']== 0)]

        num_objects = 0
        bboxes = np.array([])
        scores = np.array([])
        classes = np.array([])
        detectPersonOfInterest = pd.Series(dtype='float64')
        
        if self.initialisation :
          
          for i in detectPerson.index:
            cur_person = detectPerson.iloc[i,:]
            xmin = int(cur_person['xmin'])
            xmax = int(cur_person['xmax'])
            ymin = int(cur_person['ymin'])
            ymax = int(cur_person['ymax'])
            frame_crop = np.ascontiguousarray(frame[ymin:(ymax+1),xmin:(xmax+1),:])

            frame_crop.flags.writeable = False 

            # Make detection
            results = self.pose.process(frame_crop)
            frame_crop.flags.writeable = True

            if results.pose_landmarks:
              # Extract landmarks 
              landmarks = results.pose_landmarks.landmark
              left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
              right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
              left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
              right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

    

              if (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y):

                detectPersonOfInterest = detectPerson.iloc[i,:]          
                num_objects=1
                bboxes = np.zeros((1,4))
                scores = np.zeros(1)
                classes = np.zeros(1)
                bboxes[0,0] = int(detectPersonOfInterest['ymin'])/frame_size[0]
                bboxes[0,1] = int(detectPersonOfInterest['xmin'])/frame_size[1]
                bboxes[0,2] = int(detectPersonOfInterest['ymax'])/frame_size[0]
                bboxes[0,3] = int(detectPersonOfInterest['xmax'])/frame_size[1]
                scores[0] = np.array(float(detectPerson.iloc[0]['confidence']))
                classes[0] = np.array(0) # class person
                self.count += 1 
                if self.count > 10: 
                  self.initialisation = False
                break                     

        elif not detectPerson.empty:
        # convert data to numpy arrays and slice out unused elements
          
          num_objects = detectPerson.shape[0]

          bboxes = np.zeros((num_objects,4))
          scores = np.zeros(num_objects)
          classes = np.zeros(num_objects)

          for i in range(num_objects):
            #print('numobjects', num_objects)
            #print('i',i)
            bboxes[i,0] = int(detectPerson.iloc[i]['ymin'])/frame_size[0]
            bboxes[i,1] = int(detectPerson.iloc[i]['xmin'])/frame_size[1]
            bboxes[i,2] = int(detectPerson.iloc[i]['ymax'])/frame_size[0]
            bboxes[i,3] = int(detectPerson.iloc[i]['xmax'])/frame_size[1]
            scores[i] = np.array(float(detectPerson.iloc[i]['confidence']))
            classes[i] = np.array(0) # class person

        else:
          
          num_objects = 0
          bboxes = np.array([])
          scores = np.array([])
          classes = np.array([])

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
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

        bbox_interested = [0,0,0,0]
        class_name_interested = None
        # update tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            if self.initialisation and not detectPersonOfInterest.empty:
              self.IDofInterest = track.track_id

            if track.track_id == self.IDofInterest :
                bbox_interested = bbox
                class_name_interested = class_name

        return bbox_interested, class_name_interested

# cap = cv2.VideoCapture(0)
# detector = Detector()
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     bbox, bbox_label = detector.forward(frame)

#     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
#     result = np.asarray(frame)
#     result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     cv2.imshow("tracking", result)
#     if cv2.waitKey(10) & 0xFF == ord('q'): break
        
# cap.release()
# cv2.destroyAllWindows()

