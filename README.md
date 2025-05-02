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