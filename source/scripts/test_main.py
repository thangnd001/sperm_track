import os
import sys
from typing import Any
from pathlib import Path
from collections import Counter
import time

import cv2
import numpy as np
import torch

from scripts.sperm_classification import SpermClassification
from sperm_detection.YOLOv6.detector import SpermDetector
from sperm_tracking.yolo_tracking.boxmot import DeepOCSORT

DEBUG = True
CATEGORIES = ["Abnormal_Sperm", "Non-Sperm", "Normal_Sperm"]

class Processor(object):
    def __init__(
            self, classify_weight: str,
              detection_weight: str,
              device: str,
              yaml: str,
              img_size: int,
              half: bool,
              track_model_weights: Path, # which ReID model to use
              track_fp16: bool,
              batch_size: int,
        ) -> None:

        # sperm process init
        self.sperm_detector = SpermDetector(
            detection_weight,
            device,
            yaml,
            img_size,
            half
        )

        # tracker parameter
        self.device = device
        self.track_model_weights = track_model_weights
        self.track_fp16 = track_fp16
        self.batch_size = batch_size

    def __call__(self, video_url:str) -> Any:
        # init track
        tracker = DeepOCSORT(
            model_weights=Path(self.track_model_weights),
            # device="cuda:0",
            # device=torch.device("cuda:1"),
            device="cuda:0",
            fp16=self.track_fp16,
            det_thresh=0.1
        )

        # init statictis
        sperm_statistic = {}

        # init video
        cap = cv2.VideoCapture(video_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # write video
        if DEBUG:
            frame_width = int(cap.get(3)) # float `width`
            frame_height = int(cap.get(4))
            out = cv2.VideoWriter('outputs/result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, frameSize=(frame_width, frame_height))

        # init frame counter
        frame_counter = 0
        
        while True:
            start = time.time()
            t_check = time.time()
            ret, frame = cap.read()
            frame_counter += 1
            
            if not ret:
                break

            # Detector
            dets, img_src, crop_objs = self.sperm_detector.infer(
                                            source=frame,
                                            conf_thres=0.25,
                                            iou_thres=0.45,
                                            classes=[0],
                                            agnostic_nms=True,
                                            max_det=1000
                                        )
            
            print('TIME DETECT = ', time.time() - t_check)
            t_check = time.time()
            # Tracker
            tracks = tracker.update(dets=np.array(dets.cpu()), img=img_src)
            xyxys = tracks[:, 0:4].astype('int') # float64 to int
            ids = tracks[:, 4].astype('int') # float64 to int
            confs = tracks[:, 5]
            clss = tracks[:, 6].astype('int') # float64 to int
            inds = tracks[:, 7].astype('int') # float64 to int
            list_sperm_ids = []
            list_sperm_coords = []

            print('TIME TRACK = ', time.time() - t_check)
            t_check = time.time()

            if tracks.shape[0] != 0:
                for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                    x_cen = xyxy[0] + (xyxy[2] - xyxy[0]) / 2
                    y_cen = xyxy[1] + (xyxy[3] - xyxy[1]) / 2
                    list_sperm_ids.append(id)
                    list_sperm_coords.append((x_cen, y_cen))

                    # Draw
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]),(0, 255, 0), 1)
                    cv2.putText(frame, '{}'.format(id), (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)
                    # cv2.putText(frame, 'Type : {} : {:.2f}'.format(self.sperm_detector.class_names[cls], conf), (xyxy[0], xyxy[1] - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)
            out.write(frame)

            print('TIME WRITE TRACK = ', time.time() - t_check)
            t_check = time.time()
            
            print('TIME INFER = ', time.time() - start)

        return 0
