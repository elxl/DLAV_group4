# Change directory
import os

path = os.getcwd()
os.chdir(path + "/yolov5-deepsort")

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, path + "/yolov5-deepsort")

# load detection model
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.confidence = 0.4


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

# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
  """
  Params:
          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
  iobuf = io.BytesIO()
  # format bbox into png for return
  bbox_PIL.save(iobuf, format='png')
  # format return string
  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return bbox_bytes

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
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
def main():
  # Definition of the parameters
  max_cosine_distance = 0.05
  nn_budget = None
  nms_max_overlap = 1.0
  
  # initialize deep sort
  model_filename = 'model_data/mars-small128.pb'
  encoder = gdet.create_box_encoder(model_filename, batch_size=1)
  # calculate cosine distance metric
  metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
  # initialize tracker
  tracker = Tracker(metric, max_age=200)

  # load configuration for object detector
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)


  # Flags
  flaginfo = False


  # read in all class names from config
  class_names = utils.read_class_names(cfg.YOLO.CLASSES)

  # start streaming video from webcam
  cap = cv2.VideoCapture(0)

  # initialze bounding box to empty
  bbox = ''
  frame_num = 0
  initialisation = True
  count = 0
  IDofInterest = -1
  # while video is running
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # convert JS response to OpenCV Image    
        #frame = js_to_image(js_reply["img"])

        # grayscale image for face detection
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # create transparent overlay for bounding box
        bbox_array = np.zeros([480,640,4], dtype=np.uint8) 
        
        frame_num +=1
        #print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        start_time = time.time()

        # Detection
        result = model(frame) # inference for person
        detectPerson = result.pandas().xyxy[0] # Get the bounding box, the confidence and the class
        detectPerson = detectPerson [(detectPerson['class']== 0)]
        
        if initialisation:
          
          num_objects = 0
          bboxes = np.array([])
          scores = np.array([])
          classes = np.array([])
          detectPersonOfInterest = pd.Series(dtype='float64')
          
          for i in detectPerson.index:
            cur_person = detectPerson.iloc[i,:]
            xmin = int(cur_person['xmin'])
            xmax = int(cur_person['xmax'])
            ymin = int(cur_person['ymin'])
            ymax = int(cur_person['ymax'])
            frame_crop = np.ascontiguousarray(frame[ymin:(ymax+1),xmin:(xmax+1),:])

            frame_crop.flags.writeable = False 

            # Make detection
            results = pose.process(frame_crop)
            frame_crop.flags.writeable = True

            if results.pose_landmarks:
              # Extract landmarks 
              landmarks = results.pose_landmarks.landmark
              left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
              right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
              left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
              right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    

              if (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y):
                detectPersonOfInterest = detectPerson.iloc[i,:]         
                bbox_array = cv2.rectangle(bbox_array,(int(detectPersonOfInterest['xmin']),int(detectPersonOfInterest['ymin'])),(int(detectPersonOfInterest['xmax']),int(detectPersonOfInterest['ymax'])),(255,0,0),2)
                bbox_array = cv2.putText(bbox_array, "{} [{:.2f}]".format(detectPersonOfInterest['name'], float(detectPersonOfInterest['confidence'])),
                                  (int(detectPersonOfInterest['xmin']), int(detectPersonOfInterest['ymin']) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255,0,0), 2) 
              

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
                count += 1 
                if count > 10: 
                  initialisation = False
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
            class_name = class_names[class_indx]
            names.append(class_name)


        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
       
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            if initialisation and not detectPersonOfInterest.empty:
              IDofInterest = track.track_id
              print(IDofInterest)
            #print(track.track_id)
            if track.track_id == IDofInterest :
            # draw bbox on screen
              color = colors[int(track.track_id) % len(colors)]
              color = [i * 255 for i in color]
              cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
              cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
              cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

              # get face bounding box for overlay
              bbox_array = cv2.rectangle(bbox_array, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
              bbox_array = cv2.putText(bbox_array, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # if enable info flag then print details about each track
            if flaginfo:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        #print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        
        cv2.imshow("tracking", result)

        bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255
        # convert overlay of bbox into bytes
        bbox_bytes = bbox_to_bytes(bbox_array)
        # update bbox so next frame gets new overlay
        bbox = bbox_bytes

        if cv2.waitKey(10) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

main()