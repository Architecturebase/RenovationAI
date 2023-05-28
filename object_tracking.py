#!/usr/bin/env python
# coding: utf-8

# In[117]:


import cv2
import numpy as np
import sys
import glob

from ultralytics import YOLO
import ultralytics
import time
import torch
import os

path_video = r'C:\Users\user\reno\test_video\5.mp4'
model_name = r'C:\Users\user\reno\test_video\weights\best.pt'

class YoloDetector():
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', self.device)
        
    def load_model(self, model_name):
        if model_name:
#             model = torch.hub.load('ultralytics/yolov8', 'custom', path = model_name, force_reload = True)
#             model = torch.hub.load('ultralytics/yolov8', 'custom', path = model_name, source = 'local', force_reload = True)
            model = YOLO(model_name)
        else:
            model = torch.hub.load('ultralytics/yolov8', 'yolov8', pretrained = True)
        return model
        
    
    def score_frame(self, frame):
        
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))
#         self.model.to(self.device)
#         frame = [frame]
#         results = self.model(frame)
        results = self.model(frame)

        labels, cord, conf = results[0].boxes.xyxy[:, -1], results[0].boxes.xyxyn, results[0].boxes.conf
        cls = results[0].boxes.cls
        print(results)
        print(cls)
        print(labels, cord)
        print(conf)
        print(len(conf))
        
        return labels, cord, conf
    
    def class_to_label(self, x):
        
        return self.classes[int(x)]
    
#     def plot_boxes(self, results, frame, height, width, confidence = 0.4):
        
#         labels, cord, conf = results
#         detections = []
        
#         n = len(labels)
#         x_shape, y_shape = width, height
#         print(conf)
#         for i in range(n -1):
#             row = cord[i]
            
#             if len(conf) > 0:
#                 if conf[n] >= confidence:
#                     x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)

#                     #some_label is the label the system is supposed to process
#                     if self.class_to_label(labels[i]) == some_label:
#                         x_center = x1 + (x2 - x1)
#                         y_center = y1 + ((y2 - y1) / 2)

#                         tlwh = np.asarray([x1, y1, int(x2 - x1), int(y2 - y1)], dtype = np.float32)
#                         confidence = float(conf.item())
#                         feature = some_label

#                         detections.append(([x1, y1, int(x2-x1), int(y2 - y1)], conf.item(), some_label))
#             else:
#                 continue
                    
                    
#         return frame, detections
    def plot_boxes(self, result, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame
    
cap = cv2.VideoCapture(path_video)

# def video_upload()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

detector = YoloDetector(model_name)



from deep_sort_realtime.deepsort_tracker import DeepSort

#n_init: how many frames to determine whether the object is the exact class
#max_age: how many frames we might miss to turn the object into another label
object_tracker = DeepSort(max_age = 5, 
                         n_init = 2,
                         nms_max_overlap = 1.0,
                         max_cosine_distance = 0.3,
                         nn_budget = None,
                         override_track_class = None,
                         embedder = "mobilenet",
                         half = True,`
                         bgr = True,
                         embedder_gpu = True,
                         embedder_model_name = None,
                         embedder_wts = None,
                         polygon = False,
                         today = None)

while cap.isOpened():
    
    succes, img = cap.read()
    
    start = time.perf_counter()
    
    results = detector.score_frame(img)
#     img, detections = detector.plot_boxes(results, img, height = img.shape[0], width = img.shape[1], confidence=0.4)
#     img, detections = detector.plot_boxes(results, frame = img)
    
    tracks = object_tracker.update_tracks(detections, img)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX)
    
    end = time.perf_counter()
    totalTime = end - start
    
cap.release()

cv2.destroyAllWindows()


# In[39]:


np.version.version


# In[87]:





# In[ ]:




