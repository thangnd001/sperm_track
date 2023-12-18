import os
import sys
from typing import Any
from pathlib import Path
from collections import Counter
import time

import cv2
import numpy as np

from scripts.sperm_classification import SpermClassification
from sperm_detection.YOLOv6.detector import SpermDetector
from sperm_tracking.yolo_tracking.boxmot import DeepOCSORT

DEBUG = True
CATEGORIES = ["Abnormal_Sperm", "Non-Sperm", "Normal_Sperm"]
CATEGORIES = ['Amorphous', 'Normal', 'Pyriform', 'Tapered']
CATEGORIES = ['Normal', 'Tapered', 'Pyriform', 'Amorphous']
CATEGORIES = ['Normal', 'Abnormal']

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
        self.classificator = SpermClassification(classify_weight, device='/device:GPU:0', input_size=40)
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
            device="cuda:0",
            # device="cpu",
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
            # out = cv2.VideoWriter('outputs/result.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, frameSize=(frame_width, frame_height))

        # init frame counter
        frame_counter = 0
        
        while True:
            start = time.time()
            t_check = time.time()
            ret, frame = cap.read()
            frame_counter += 1
            
            # if not ret:
            #     break

            # Detector
            try:
                dets, img_src, crop_objs = self.sperm_detector.infer(
                                                source=frame,
                                                conf_thres=0.25,
                                                iou_thres=0.45,
                                                classes=[0],
                                                agnostic_nms=True,
                                                max_det=1000
                                            )
            except:
                break
            
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
            

            print('TIME WRITE TRACK = ', time.time() - t_check)
            t_check = time.time()

            # Sperm classification
            list_sperm_types = self.classificator(crop_objs, self.batch_size)
            for index, (det_ind, sperm_type) in enumerate(zip(inds, list_sperm_types)):
                sperm_id = list_sperm_ids[det_ind]
                coord = list_sperm_coords[det_ind]
                if sperm_id not in sperm_statistic:
                    sperm_statistic[sperm_id] = {
                        'type': [sperm_type], 
                        'coord_track': [coord], 
                        'frame_track':[frame_counter, frame_counter],
                        'list_frame_time': [frame_counter]
                        }
                else:
                    sperm_statistic[sperm_id]['type'].append(sperm_type)
                    sperm_statistic[sperm_id]['coord_track'].append(coord)
                    sperm_statistic[sperm_id]['frame_track'].pop(-1)
                    sperm_statistic[sperm_id]['frame_track'].append(frame_counter)
                    sperm_statistic[sperm_id]['list_frame_time'].append(frame_counter)
                
                # Draw
                xyxy = xyxys[index]
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]),(0, 255, 0) if sperm_type == 0 else (0, 0, 255), 1)
                cv2.putText(frame, '{}'.format(sperm_id), (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)
                # cv2.putText(frame, 'Type : {} : {:.2f}'.format(self.sperm_detector.class_names[cls], conf), (xyxy[0], xyxy[1] - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)
            out.write(frame)
            
            print('TIME INFER = ', time.time() - start)

        return self.statistic(sperm_statistic, fps, frame_height, frame_width)
        
    def stop(self):
        pass

    def statistic(self, sperm_statistic: dict, fps: int, frame_height: int, frame_width: int):
        """
        return: statistical analysis
        """
        def cal_speed(list_coord:list, list_frame_time:list, num_frame: int, fps:int, ratio_h: float, ratio_w: float):
            """
            return the distance traveled
            """
            if len(list_coord) == 0 or num_frame == 0:
                return 0
    
            current_coord = convert_pixel2micro(list_coord[0], ratio_h, ratio_w)
            current_frame_time = list_frame_time[0]
            distance = 0
            for frame_time, coord in zip(list_frame_time, list_coord):
                if frame_time - current_frame_time < 10:
                    continue
                else:
                    coord_convert = convert_pixel2micro(coord, ratio_h, ratio_w)
                    distance += np.linalg.norm(np.array(coord_convert, dtype=float) - np.array(current_coord, dtype=float))
                    current_coord = coord_convert
                    current_frame_time = frame_time
            # print('DISTANCE = ', distance)/
            return distance / (num_frame / fps)

        def get_type(list_type: list):
            """
            return type of sperm
            """
            counter = Counter(list_type)
            
            return list(counter.keys())[0]
        
        def convert_pixel2micro(coord: list, h_ratio: float, w_ratio: float):
            coord_copy = list(coord).copy()
            coord_copy[0] *= w_ratio
            coord_copy[1] *= h_ratio
            
            return coord_copy
        
        # def convert_pixel2micro(x:int, y:int, ratio:float):
        #     # M_obj = 0 # Objective Lens Magnification
        #     # M_ocul = 0 # Ocular Lens Magnification
        #     # M_cam = 0 # Camera Lens Magnification
        #     # S_sensor = 0 # Camera Sensor Format
        #     # W_sensor = 0 # Camera Sensor Width
        #     # H_sensor = 0 # Camera Sensor Height
        #     W_sample = 480
        #     H_SAMPLE = 640
             
        # Convert micromet
        W_sample = 480
        H_SAMPLE = 640
        ratio_w = W_sample / frame_width
        ratio_h = H_SAMPLE / frame_height
        
        # Process
        list_sperm_statistic = {}
        for sperm_id, value in sperm_statistic.items():
            speed = cal_speed(
                list_coord=value['coord_track'],
                list_frame_time=value['list_frame_time'],
                num_frame=value['frame_track'][-1] - value['frame_track'][0], 
                fps=fps, 
                ratio_h=ratio_h, 
                ratio_w=ratio_w
            )
            sperm_type = CATEGORIES[get_type(value['type'])]
            result_info = {'type': sperm_type, 'speed': speed}
            list_sperm_statistic[sperm_id] = result_info

        speed_result = []
        statistic_final_Result = {}
        for type_sperm in CATEGORIES:
            statistic_final_Result[type_sperm] = 0
        for sperm_id, value in list_sperm_statistic.items():
            statistic_final_Result[value['type']] += 1
            speed_result.append(int(value['speed']))
        
        final_result = [speed_result, statistic_final_Result]
                   
        return final_result

        