import os
import argparse
from collections import Counter
import json

import altair as alt
import gradio as gr
import numpy as np
import pandas as pd
import gradio as gr

from scripts.main import Processor


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify_weight',default='sperm_classification/runs/training/HuSHeM_dataset/version-0.0/model-save/model-HuSHeM_dataset-version-0.0.h5', type=str)
    parser.add_argument('--detection_weight', default='sperm_detection/weights/best_ckpt.pt', type=str)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--yaml', default='sperm_detection/YOLOv6/data/miamia-sperm.yaml', type=str)
    parser.add_argument('--img_size', default=[640, 640], type=str)
    parser.add_argument('--half', default=False, type=str)
    parser.add_argument('--track_model_weights', default='sperm_tracking/weigths/osnet_x0_25_msmt17.pt', type=str)
    parser.add_argument('--track_fp16', default=False, type=str)
    parser.add_argument('--batch_size', default=96, type=str)

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


statistic_global = {}
speed_statistic = []


def eval(video):
    # global statistic_global
    # global speed_statistic
    # output = processor(video)
    
    # statistic_global = output[1]
    # speed_statistic = output[0]
    print("VIDEO = ", video)    
    return f'outputs/result.avi'



def stop():
    pass

def make_plot(plot_type):
    if plot_type == "Types of sperm":
        if len(statistic_global) == 0 :
            return None
        
        pts = alt.selection(type="single", encodings=['x'])
        soure_data = {"Type of sperm": [], "Total sperm": []}
        for key, value in statistic_global.items():
            soure_data["Type of sperm"].append(key)
            soure_data["Total sperm"].append(value)

        with open('Type of sperm.json', 'w') as tf:
            json.dump(soure_data, tf, indent=2)
        
        source = pd.DataFrame(soure_data)
        
        bar = alt.Chart(source).mark_bar().encode(
            x='Type of sperm',
            y='Total sperm',
            color=alt.condition(pts, alt.ColorValue("steelblue"), alt.ColorValue("grey"))
        ).properties(
            width=550,
            height=200
        ).add_selection(pts)
        
        return bar

    elif plot_type == "Sperm velocity":
        data = dict(Counter(speed_statistic))
        pts = alt.selection(type="single", encodings=['x'])
        data_convert = {"Sperm velocity": [], "Total sperm":[]}
        for key, value in data.items():
            data_convert["Sperm velocity"].append(key)
            data_convert["Total sperm"].append(value)

        with open('Sperm velocity.json', 'w') as tf:
            json.dump(data_convert, tf, indent=2)

        source = pd.DataFrame(data_convert)

        bar = alt.Chart(source).mark_bar().encode(
            x='Sperm velocity',
            y='Total sperm',
            color=alt.condition(pts, alt.ColorValue("red"), alt.ColorValue("grey"))
        ).properties(
            width=550,
            height=200
        ).add_selection(pts)
        
        return bar

# def 

def main_ui():
    
    with gr.Blocks() as demo:
        with gr.Blocks():
            with gr.Row():
                label = gr.Label("Sperm Tracking", show_label=False, scale=25, container=False)
        # Process video
        with gr.Blocks() as process_video:
            with gr.Row():
                video_upload = gr.Video(label='Input')
                video_processed = gr.Video(label='Ouput1')
                
            with gr.Row():
                submit_btn = gr.Button(value='Submit', variant='primary')
                # clear_btn = gr.Button(value='Clear', variant='primary')
            
        # Bar result
        with gr.Blocks() as statistic_result:
            button = gr.Radio(label="Plot type",
                            choices=['Types of sperm', "Sperm velocity"], value='Types of sperm')
            plot = gr.Plot(label="Plot", show_label=True, container=True)
            button.change(make_plot, inputs=button, outputs=[plot])
        
        # Process event
        submit_btn.click(
            fn=eval,
            inputs=video_upload,
            outputs=[video_processed],
            show_progress=True
        )
        
    return demo


if __name__ == "__main__":
    demo = main_ui()
    demo.launch(share=True)



