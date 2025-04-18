import os
import argparse
from collections import Counter
import json
import requests
from io import BytesIO
from PIL import Image, ImageOps  # Add this import for background removal

import altair as alt
import gradio as gr
import numpy as np
import pandas as pd
import cv2

from scripts.main import Processor


# Directory setup
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
LOGO_PATH = os.path.join(ASSETS_DIR, "haui_logo_updated.jpg")
MEDICAL_ICON_PATH = os.path.join(ASSETS_DIR, "medical_icon.png")

# Update the logo path to the new HAUI logo
def ensure_logo_exists():
    if not os.path.exists(LOGO_PATH):
        try:
            # Save the provided HAUI logo
            logo_url = "path/to/new/haui_logo.jpg"  # Replace with the actual path to the provided logo
            response = requests.get(logo_url)
            img = Image.open(BytesIO(response.content)).convert("RGBA")
            
            # Remove white background
            data = np.array(img)
            red, green, blue, alpha = data.T
            white_areas = (red == 255) & (green == 255) & (blue == 255)
            data[..., :-1][white_areas.T] = (0, 0, 0)  # Set white areas to black
            data[..., -1][white_areas.T] = 0  # Set alpha to 0 for transparency
            img = Image.fromarray(data)
            
            img.save(LOGO_PATH)
            print(f"Logo downloaded, processed, and saved to {LOGO_PATH}")
        except Exception as e:
            print(f"Error downloading or processing logo: {e}")
    return LOGO_PATH

# Download the medical icon if it doesn't exist
def ensure_medical_icon_exists():
    if not os.path.exists(MEDICAL_ICON_PATH):
        try:
            # Save the caduceus medical symbol
            response = requests.get("https://static.wixstatic.com/media/9d8ed5_f0d7ea50fd804ba9a93d9f34029d8695~mv2.png/v1/fill/w_500,h_500,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/9d8ed5_f0d7ea50fd804ba9a93d9f34029d8695~mv2.png")
            img = Image.open(BytesIO(response.content))
            img.save(MEDICAL_ICON_PATH)
            print(f"Medical icon downloaded and saved to {MEDICAL_ICON_PATH}")
        except Exception as e:
            print(f"Error downloading medical icon: {e}")
    return MEDICAL_ICON_PATH

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
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    try:
        # Process the video
        output = processor(video)
        
        # Store statistics
        statistic_global = output[1]
        speed_statistic = output[0]
        
        # Get output path
        output_path = 'outputs/result.avi'
        mp4_path = 'outputs/result.mp4'
        
        # Check if file exists
        if os.path.exists(output_path):
            # Convert AVI to MP4 for better compatibility with web browsers
            try:
                # Read the AVI file
                cap = cv2.VideoCapture(output_path)
                if not cap.isOpened():
                    print(f"Error: Could not open {output_path}")
                    return output_path  # Return original path as fallback
                
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Create MP4 writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
                
                # Process frame by frame
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                
                # Release resources
                cap.release()
                out.release()
                
                print(f"Successfully converted video to MP4: {mp4_path}")
                return mp4_path
                
            except Exception as e:
                print(f"Error converting video to MP4: {str(e)}")
                return output_path  # Return original path as fallback
        else:
            print(f"Warning: Output file not found at {output_path}")
            return None
            
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None

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
    # Make sure we have the logo and medical icon
    logo_path = ensure_logo_exists()
    medical_icon_path = ensure_medical_icon_exists()
    
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
            background-color: #1e40af; /* Changed to blue */
            color: white; /* Changed to white */
        }
        .header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 30px;
            padding: 15px;
            background-color: #2563eb; /* Changed to lighter blue */
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            color: white; /* Changed to white */
        }
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .logo-container img {
            object-fit: contain;
            max-height: 120px;
            max-width: 100%;
        }
        .title-container {
            text-align: center;
        }
        .app-title {
            font-size: 32px;
            font-weight: bold;
            margin: 0;
            color: white !important; /* Changed to white */
            text-align: center;
        }
        .app-subtitle {
            font-size: 18px;
            margin: 10px 0 0 0;
            color: white !important; /* Changed to white */
            text-align: center;
        }
        .container {
            margin: 20px 0;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            background-color: #3b82f6; /* Changed to medium blue */
            color: white; /* Changed to white */
        }
        .section-title {
            color: white; /* Changed to white */
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 15px;
            text-align: center;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 14px;
            padding: 15px;
            background-color: #2563eb; /* Changed to lighter blue */
            color: white !important; /* Ensure footer text is white */
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .footer p {
            color: white !important; /* Explicitly set paragraph text color */
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
                with gr.Column(scale=1, min_width=120, elem_classes="logo-container"):
                    gr.Image(logo_path, show_label=False, show_download_button=False, height=120)
                with gr.Column(scale=3, elem_classes="title-container"):
                    gr.HTML(
                        """
                        <div>
                            <h1 class="app-title">Human Sperm Analysis Tool</h1>
                            <p class="app-subtitle">Hanoi University of Industry</p>
                        </div>
                        """
                    )
                with gr.Column(scale=1, min_width=120, elem_classes="logo-container"):
                    gr.Image(medical_icon_path, show_label=False, show_download_button=False, height=120)
            
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
                <div class="footer" style="color: white !important; text-align: center;">
                    <p style="color: white !important;">© 2025 Hanoi University of Industry - Biomedical Imaging Research Group. All rights reserved.</p>
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