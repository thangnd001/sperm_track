import os
import argparse

import gradio as gr

from source.scripts.main import Processor


processor = Processor()

def ui():
    def eval():
        pass

    def stop():
        pass

    with gr.Blocks as demo:
        with gr.Row():
            video = gr.Video()
            output = gr.Image(label='Statistic')
        with gr.Row():
            eval_btn = gr.Button('Evaluate')
            stop_btn = gr.Button('Stop')
        
        eval_btn.click(
            fn=eval,
            inputs=[video],
            outputs=[output]
            )
        
        stop_btn.click(
            fn=stop,
            inputs=[],
            outputs=[]
        )
        
    return demo

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify_weight',default=None, type=str)
    parser.add_argument('--detection_weight', default=None, type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--yaml', default=None, type=str)
    parser.add_argument('--img_size', default=None, type=str)
    parser.add_argument('--half', default=None, type=str)
    parser.add_argument('--track_model_weights', default=None, type=str)
    parser.add_argument('--track_fp16', default=None, type=str)
    parser.add_argument('--batch_size', default=None, type=str)

    args = parser.parse_args()
    return args


