import os
from typing import Any
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

from sperm_classification import SpermClassification
from sperm_detection.YOLOv6.detector import SpermDetector
from sperm_tracking.yolo_tracking.boxmot import DeepOCSORT


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
        self.classificator = SpermClassification(classify_weight)
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
            device=self.device,
            fp16=self.track_fp16
        )

        # init statictis
        sperm_statistic = {}

        # init video
        cap = cv2.VideoCapture(video_url)
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()

            # Detector
            dets, img_src, crop_objs = self.sperm_detector.infer(
                                            source=frame,
                                            conf_thres=0.25,
                                            iou_thres=0.45,
                                            classes=None,
                                            agnostic_nms=True,
                                            max_det=1000
                                        )
            

            # Tracker
            tracks = tracker.update(dets=dets, img=img_src)
            xyxys = tracks[:, 0:4].astype('int') # float64 to int
            ids = tracks[:, 4].astype('int') # float64 to int
            confs = tracks[:, 5]
            clss = tracks[:, 6].astype('int') # float64 to int
            inds = tracks[:, 7].astype('int') # float64 to int
            list_sperm_ids = []
            list_sperm_coords = []

            if tracks.shape[0] != 0:
                for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                    x_cen = xyxy[0] + (xyxy[2] - xyxy[0]) / 2
                    y_cen = xyxy[1] + (xyxy[3] - xyxy[1]) / 2
                    list_sperm_ids.append(id)
                    list_sperm_coords.append((x_cen, y_cen))

                    # Draw
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]),(0, 255, 0), 1)
                    cv2.putText(frame, '{}'.format(id), (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 1), 2)
                    cv2.putText(frame, 'Type : {} : {:.2f}'.format(cls, conf), (xyxy[0], xyxy[1] - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)

            # Sperm classification
            list_sperm_types = self.classificator(crop_objs, self.batch_size)
            for sperm_id, sperm_type, coord in zip(list_sperm_ids, list_sperm_types, list_sperm_coords):
                if sperm_id not in sperm_statistic:
                    sperm_statistic[sperm_id] = {'type': [sperm_type], 'coord track': [coord]}
                else:
                    sperm_statistic[sperm_id]['type'].append(sperm_type)
                    sperm_statistic[sperm_id]['coord_track'].append(coord)

        return self.statistic(sperm_statistic, fps)
        
    def stop(self):
        pass

    def statistic(self, sperm_statistic: dict, fps: int):
        """
        return: statistical analysis
        """
        def cal_distance(list_coord):
            """
            return the distance traveled
            """
            current_coord = list_coord[0]
            distance = 0
            for coord in list_coord:
                distance += np.linalg.norm(coord - current_coord)
                current_coord = coord

            return distance

        def get_type(list_type: list):
            """
            return type of sperm
            """
            counter = Counter(list_type)
            
            return counter.keys()[0]

        list_sperm_statistic = [] 
        for sperm_id, value in sperm_statistic.items():
            distance = cal_distance(value['coord'])
            sperm_type = get_type(value['type'])
            list_sperm_statistic.append(sperm_type)
        
        return list_sperm_statistic

        