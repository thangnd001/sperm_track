import os
import argparse
from collections import Counter
import json
import requests
from io import BytesIO
from PIL import Image

import altair as alt
import gradio as gr
import numpy as np
import pandas as pd

from scripts.main import Processor


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# Create a directory for assets if it doesn't exist
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
LOGO_PATH = os.path.join(ASSETS_DIR, "haui_logo.jpg")

# Download the logo if it doesn't exist
def ensure_logo_exists():
    if not os.path.exists(LOGO_PATH):
        try:
            logo_url = "https://inkythuatso.com/uploads/thumbnails/800/2021/12/logo-dai-hoc-cong-nghiep-ha-noi-inkythuatso-01-21-16-44-16.jpg"
            response = requests.get(logo_url)
            img = Image.open(BytesIO(response.content))
            img.save(LOGO_PATH)
            print(f"Logo downloaded and saved to {LOGO_PATH}")
        except Exception as e:
            print(f"Error downloading logo: {e}")
    return LOGO_PATH


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
    
# Initialize processor
processor = Processor(*get_args())
statistic_global = {}
speed_statistic = []


def eval(video):
    global statistic_global
    global speed_statistic
    output = processor(video)
    
    statistic_global = output[1]
    speed_statistic = output[0]
    print("VIDEO = ", video)    
    return f'outputs/result.avi'


def make_plot(plot_type):
    if plot_type == "Types of sperm":
        pts = alt.selection(type="single", encodings=['x'])

        with open('Type of sperm.json', 'r') as tf:
            soure_data = json.load(tf)
        
        source = pd.DataFrame(soure_data)
        
        bar = alt.Chart(source).mark_bar().encode(
            x=alt.X('Type of sperm', title='Type of Sperm'),
            y=alt.Y('Total sperm', title='Total Sperm Count'),
            color=alt.condition(
                pts, 
                alt.ColorValue("#1e40af"), 
                alt.ColorValue("lightgray")
            ),
            tooltip=['Type of sperm', 'Total sperm']
        ).properties(
            width=550,
            height=300,
            title='Sperm Type Distribution'
        ).configure_title(
            fontSize=16,
            font='Arial',
            anchor='middle',
            color='#1e3a8a'
        ).add_selection(pts)
        
        return bar, source

    elif plot_type == "Sperm velocity":
        pts = alt.selection(type="single", encodings=['x'])

        with open('Sperm velocity.json', 'r') as tf:
            data_convert = json.load(tf)

        new_data = {
            "(-1,20]": 0,
            "(20,40]": 0,
            "(40,∞)": 0,
        }
        for v, t in zip(data_convert['Sperm velocity(Micrometre/s)'], data_convert['Total sperm']):
            if v >=0 and v <= 20:
                new_data['(-1,20]'] += t
            elif v > 20 and v <= 40:
                new_data['(20,40]'] += t
            else:
                new_data['(40,∞)'] += t
        
        data_convert = {"Sperm velocity(Micrometre/s)": [], "Total sperm":[]}
        for key, value in new_data.items():
            data_convert["Sperm velocity(Micrometre/s)"].append(key)
            data_convert["Total sperm"].append(value)
        
        source = pd.DataFrame(data_convert)

        bar = alt.Chart(source).mark_bar().encode(
            x=alt.X('Sperm velocity(Micrometre/s)', title='Velocity Range (μm/s)'),
            y=alt.Y('Total sperm', title='Number of Sperm'),
            color=alt.condition(
                pts, 
                alt.ColorValue("#0369a1"), 
                alt.ColorValue("lightgray")
            ),
            tooltip=['Sperm velocity(Micrometre/s)', 'Total sperm']
        ).properties(
            width=550,
            height=300,
            title='Sperm Velocity Distribution'
        ).configure_title(
            fontSize=16,
            font='Arial',
            anchor='middle',
            color='#0c4a6e'
        ).add_selection(pts)
        
        return bar, source


def main_ui():
    # Make sure we have the logo
    logo_path = ensure_logo_exists()
    
    # Create custom theme
    theme = gr.themes.Monochrome(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        body_background_fill="#f8fafc",
        body_text_color="#1e293b",
        body_text_size="16px",
        button_primary_background_fill="#1e40af",
        button_primary_background_fill_hover="#2563eb",
        button_secondary_background_fill="#e2e8f0",
        button_secondary_background_fill_hover="#cbd5e1",
        block_label_background_fill="#f8fafc",
        block_shadow="0px 4px 6px rgba(0, 0, 0, 0.1)",
        block_title_text_weight="600",
        container_radius="12px",
        panel_background_fill="#ffffff"
    )
    
    with gr.Blocks(theme=theme, css="""
        #app-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 30px;
            padding: 15px;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .logo-container {
            margin-right: 25px;
            display: flex;
            justify-content: center;
        }
        .title-container {
            text-align: center;
        }
        .app-title {
            font-size: 32px;
            font-weight: bold;
            margin: 0;
            color: #1e3a8a;
            text-align: center;
        }
        .app-subtitle {
            font-size: 18px;
            margin: 10px 0 0 0;
            color: #475569;
            text-align: center;
        }
        .container {
            margin: 20px 0;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: white;
        }
        .section-title {
            color: #1e3a8a;
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 15px;
            text-align: center;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 14px;
            color: #64748b;
            padding: 15px;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .button-group {
            display: flex;
            justify-content: center;
            gap: 15px;
        }
    """) as demo:
        with gr.Column(elem_id="app-container"):
            # Header with logo and title
            with gr.Row(elem_classes="header-container"):
                with gr.Column(scale=1, min_width=150, elem_classes="logo-container"):
                    gr.Image(logo_path, show_label=False, height=120)
                with gr.Column(scale=3, elem_classes="title-container"):
                    gr.HTML(
                        """
                        <div>
                            <h1 class="app-title">Human Sperm Analysis Tool</h1>
                            <p class="app-subtitle">Hanoi University of Industry - Research Laboratory</p>
                        </div>
                        """
                    )
            
            # Process video section
            with gr.Box(elem_classes="container"):
                gr.HTML("<h2 class='section-title'>Video Analysis</h2>")
                with gr.Row():
                    with gr.Column(scale=1):
                        video_upload = gr.Video(label="Upload Video Sample")
                        with gr.Row(elem_classes="button-group"):
                            submit_btn = gr.Button(value="Analyze Video", variant="primary", size="lg")
                            clear_btn = gr.Button(value="Clear", variant="secondary", size="lg")
                        
                    with gr.Column(scale=1):
                        video_processed = gr.Video(label="Analysis Result")
            
            # Analysis and statistics section
            with gr.Box(elem_classes="container"):
                gr.HTML("<h2 class='section-title'>Statistical Analysis</h2>")
                with gr.Tabs() as tabs:
                    with gr.TabItem("Visualization"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                button = gr.Radio(
                                    label="Data Visualization Type",
                                    choices=['Types of sperm', "Sperm velocity"], 
                                    value='Types of sperm',
                                    info="Select the type of data to visualize"
                                )
                        
                        with gr.Row():
                            with gr.Column(scale=3, min_width=600):
                                plot = gr.Plot(label="", show_label=False)
                            with gr.Column(scale=2):
                                statistic_data = gr.DataFrame(label='Detailed Statistics')
                    
                    with gr.TabItem("About"):
                        gr.Markdown("""
                        ## Human Sperm Analysis Tool
                        
                        This advanced application uses computer vision and machine learning to analyze human sperm samples:
                        
                        * **Detection**: Identifies individual sperm cells in video samples
                        * **Tracking**: Monitors movement patterns and trajectories
                        * **Classification**: Categorizes sperm based on morphology and motility
                        * **Analysis**: Calculates velocity, linearity, and other key metrics
                        * **Reporting**: Generates comprehensive statistical reports
                        
                        ### Technology Stack
                        
                        * **Detection Engine**: YOLOv6 (You Only Look Once)
                        * **Tracking System**: Deep SORT (Simple Online and Realtime Tracking)
                        * **Classification Model**: Custom CNN architecture
                        * **Data Visualization**: Altair & Gradio
                        
                        ### Research Team
                        
                        Developed by the Biomedical Imaging Research Laboratory at Hanoi University of Industry.
                        """)
            
            # Footer
            gr.HTML(
                """
                <div class="footer">
                    <p>© 2025 Hanoi University of Industry - Biomedical Imaging Research Group. All rights reserved.</p>
                </div>
                """
            )

            # Event handlers
            submit_btn.click(
                fn=eval,
                inputs=video_upload,
                outputs=video_processed,
                show_progress=True
            )

            clear_btn.click(
                fn=lambda: None,
                inputs=[],
                outputs=video_upload
            )

            button.change(make_plot, inputs=button, outputs=[plot, statistic_data])
        
    return demo


if __name__ == "__main__":
    demo = main_ui()
    demo.launch(share=True)