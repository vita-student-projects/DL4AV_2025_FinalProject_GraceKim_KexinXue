DLAV milestone 3

Group: 
Kexin Xue(GithubID: kxk-coding)not yet added to Github project
Grace Eunhyeong Kim(GithubID: kimgracy)

 Augmented Camera Dataset with YOLO Segmentation

This model introduces a starter deep learning pipeline by integrating **YOLO segmentation** to generate labeled training data from augmented camera input. The updated model leverages these high-quality labels for more accurate object detection and segmentation.

## Key Improvements from Starter Code

- **YOLO Segmentation Integration:** 
  - The model uses a YOLO-based segmentation model (e.g., YOLOv8-seg) to automatically label images from augmented camera data.
  - Enables automatic annotation for supervised training without manually labeled data.

- **Augmented Camera Data:**
  - Data from multiple camera perspectives or modified viewpoints is processed and used for training.
  - Augmentations improve robustness and generalization of the model.

- **Training Pipeline Updates:**
  - The segmentation outputs are used to create masks and bounding boxes in the required format for training.
  - The final model is fine-tuned on this enriched dataset to improve performance in downstream tasks.

## User Instruction

### 1. Set Up Environment

Install required dependencies:

```bash
pip install -r requirements.txt
```

Make sure to include YOLO dependencies, e.g.:

```bash
pip install ultralytics
```

### 2. Prepare YOLO Segmentation Model

Download a pre-trained YOLO segmentation model (e.g., YOLOv8-seg):

```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # Replace with preferred variant
```

### 3. Generate Labels from Camera Data

Use YOLO to generate masks/labels from raw camera frames:

```python
results = model('path/to/image_or_video_frame.jpg')
results[0].save_crop(save_dir='labeled_data/')  # Save cropped regions and labels
```

These outputs are used to build a dataset with segmentation masks or bounding boxes.

### 4. Train the Model

Use the labeled dataset to train your custom model. Make sure the labels are in the format expected by your training loop.

## 📁 Project Structure

```
DLAV_Phase3/
│
├── DLAV_Phase3.ipynb       # Notebook with model pipeline and YOLO integration
├── labeled_data/           # YOLO-generated labeled data
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Results

The limited dataset has been leading to overfitting of more complex architectures, this pre step of data augumentation helped with better processing the data.YOLO provides real-time object detection or segmentation, identifying dynamic agents like pedestrians, cyclists, or vehicles. This gives your trajectory model: Clear bounding boxes or masks of objects; Better awareness of which entities to track and predict; Contextual awareness, like who is near whom and possible interactions. 
Without YOLO: The model may rely on pixel-based heuristics or manual labels to locate agents.
With YOLO: Agent positions are accurately and automatically localized, giving the trajectory model precise initial positions and movement vectors.

