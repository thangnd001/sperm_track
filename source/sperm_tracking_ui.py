import os
import argparse

import gradio as gr

from scripts.main import Processor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify_weight',default='sperm_classification/model/smids_mobiv2.h5', type=str)
    parser.add_argument('--detection_weight', default='sperm_detection/weights/yolov6t_segrelu.pt', type=str)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--yaml', default='sperm_detection/YOLOv6/data/miamia-sperm.yaml', type=str)
    parser.add_argument('--img_size', default=[640, 640], type=str)
    parser.add_argument('--half', default=False, type=str)
    parser.add_argument('--track_model_weights', default='sperm_tracking/weigths/osnet_x0_25_msmt17.pt', type=str)
    parser.add_argument('--track_fp16', default=False, type=str)
    parser.add_argument('--batch_size', default=1, type=str)

    args = parser.parse_args()
    classify_weight = args.classify_weight
    detection_weight = args.detection_weight
    device = args.device
    yaml = args.yaml
    img_size = args.img_size
    half = args.half
    track_model_weights = args.track_model_weights
    track_fp16 = args.track_fp16
    batch_size = args.batch_size

    return (
        classify_weight, 
        detection_weight, 
        device,
        yaml,
        img_size,
        half,
        track_model_weights,
        track_fp16,
        batch_size
        )
    
# process

processor = Processor(*get_args())
def ui():
    def eval(video):
        output = processor(video)
        
        print(output)
        
        return 'outputs/result.avi'

    def stop():
        pass

    with gr.Blocks() as demo:
        with gr.Row():
            video = gr.Video()
            output = gr.Video(label='Statistic')
        with gr.Row():
            eval_btn = gr.Button('Evaluate')
            stop_btn = gr.Button('Stop')
        
        eval_btn.click(
            fn=eval,
            inputs=[video],
            outputs=output,
            show_progress=True
            )
        
        stop_btn.click(
            fn=stop,
            inputs=[],
            outputs=[]
        )
        
    return demo

