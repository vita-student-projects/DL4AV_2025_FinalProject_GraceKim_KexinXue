# DLAV 2025 : Final Project

**Author**: Grace Eunhyeong Kim and Kexin Xue
**Course**: Deep Learning for Autonomous Vehicles

## MileStone 1: Basic End-to-End Planner

For project, an end-to-end model that predicts future trajectories is defined.
It is based on the following inputs :

- Camera RGB image
- Driving Command
- Vehicle's motion history (sdc_history_feature)


### Model & Training Method

- **Model Architecture**:
  1. *ResNet-50 Backbone*:
     - ResNet-50 model is used as a backbone for extracting features from image.
     - `self.cnn` consists of all layers of ResNet except for the last two.
     - `self.avgpool` is used to apply adaptive average pooling in order to reduce spatial dimension.

  2. *Decoder*:
     - `self.decoder` is used to predict future path using visual features and historical trajectories.
     - It consists of two fc layers and one ReLU activation function.


- **Forward Pass**:
  1. *Image Processing*:
     - `x = self.cnn(camera)`, `x = self.avgpool(x)`, `visual_features = torch.flatten(x, 1)`
     - ResNet-50 is used as a backbone to extract features from the image. Adaptive average pooling is used in order to reduce the spatial dimension, and the extracted features are flattened into a 1D vector.

  2. *Historical Trajectory Processing*:
     - `history_flat = history.reshape(history.size(0), -1)`
     - Historical Trajectory is processed by simply flattening the historical trajectory, so that it can be used for further processing.

  3. *Feature Concatenation*:
     - `combined = torch.cat((visual_features, history_flat), dim=1)`
     - The visual features and historical trajectories are concatenated into a single vector.

  4. *Future Path Prediction*:
     - `future = self.decoder(combined)`, `future = future.view(batch_size, -1, history.size(2))`
     - The combined input is passed through the decoder, and the predicted future path is reshaped into its original dimensions.


### Used Hyperparameters
- `batch_size` of 32
- `learning_rate` of 1e-3
- `num_epochs` of 70
- Adam optimizer


### How to train the model
```
model = DrivingPlanner()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

train(model, train_loader, val_loader, optimizer, logger, num_epochs=70)
```


### How to run the model
```
model = DrivingPlanner()

# camera = RGB image input
# history = Vehicle's motion history

with torch.no_grad():
    pred_future = model(camera, history)
```


## DLAV milestone 3

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

