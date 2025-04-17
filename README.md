# User Guide for Human Sperm Analysis Tool

This guide explains how to set up and use the Human Sperm Analysis Tool for analyzing sperm samples using the `sperm_tracking_ui.py` application.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Using the Interface](#using-the-interface)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements
- **Processor**: Intel Core i5 (8th generation or newer) or AMD Ryzen 5 (2000 series or newer)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (at least 4GB VRAM) recommended
- **Storage**: 10GB free disk space

### Software Requirements
- **Operating System**: Linux (Ubuntu 18.04 or newer recommended)
- **Python**: Version 3.8 or higher
- **CUDA Toolkit**: Version 10.2 or higher (for GPU acceleration)
- **Web Browser**: Chrome, Firefox, or Edge (latest versions)

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone 
   



   
   cd sperm_strack
   git checkout dev
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Models**
   - Detection model: Place in `sperm_detection/weights/best_ckpt.pt`
   - Classification model: Place in `sperm_classification/runs/training/HuSHeM_dataset/version-0.0/model-save/model-HuSHeM_dataset-version-0.0.h5`
   - Tracking model: Place in `sperm_tracking/weigths/osnet_x0_25_msmt17.pt`

---

## Configuration

The application uses default parameters, but you can customize them by editing the `sperm_tracking_ui.py` file or passing arguments via the command line.

### Key Parameters
| Parameter              | Default Value                                              | Description                              |
|------------------------|----------------------------------------------------------|------------------------------------------|
| `--classify_weight`    | `sperm_classification/.../model-HuSHeM_dataset-version-0.0.h5` | Path to the classification model         |
| `--detection_weight`   | `sperm_detection/weights/best_ckpt.pt`                    | Path to the detection model              |
| `--device`             | `0`                                                      | GPU device ID (`-1` for CPU)             |
| `--img_size`           | `[640, 640]`                                             | Input image dimensions                   |
| `--batch_size`         | `96`                                                     | Batch size for processing                |

---

## Running the Application

1. **Navigate to the Source Directory**
   ```bash
   cd source
   ```

2. **Launch the Application**
   ```bash
   python app.py
   ```

3. **Access the Interface**
   - Open the URL displayed in the terminal (e.g., `http://127.0.0.1:7860/`) in your web browser.

---

## Using the Interface

### Step 1: Upload a Video
- Drag and drop a video file into the "Input" section or click to browse your file system.
- Supported formats: MP4, AVI, MOV, MKV.

### Step 2: Process the Video
- Click the **Analyze Video** button to start the analysis.
- The application will process the video and generate results.

### Step 3: View Results
- **Processed Video**: The output video with tracking overlays will appear in the "Output" section.
- **Statistics**:
  - Select "Types of sperm" or "Sperm velocity" to view corresponding charts.
  - The data table will display detailed statistics.

---

## Troubleshooting

### Common Issues

#### 1. **Video Upload Fails**
- Ensure the video format is supported.
- Check file size (should be under 200MB).

#### 2. **CUDA Out of Memory**
- Reduce batch size:
  ```bash
  python app.py --batch_size 32
  ```
- Use CPU mode:
  ```bash
  python app.py --device -1
  ```

#### 3. **Missing Models**
- Verify that all required models are downloaded and placed in the correct directories.

#### 4. **UI Not Loading**
- Ensure you are using a modern browser.
- Check for errors in the terminal or browser console.

---

For further assistance, please contact the development team or refer to the project's GitHub repository.