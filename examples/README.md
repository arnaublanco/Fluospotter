This folder contains example notebooks and scripts demonstrating how to use Fluospotter. The following examples are provided:

### 1. `nuclei_segmentation_train.ipynb`
- **Description**: This notebook demonstrates how to train a segmentation model on cell nuclei data using the `fluospotter` library.
- **Key Features**:
  - Loading and preprocessing of training data.
  - Model training with various configurations.
  - Evaluation and visualization of results.

### 2. `nuclei_segmentation_prediction.ipynb`
- **Description**: This notebook shows how to use a trained model to predict segmentations on new images. It includes chunk-wise segmentation for large volumes.
- **Key Features**:
  - Loading a pre-trained model.
  - Predicting segmentations on test data.
  - Handling large image volumes with chunk processing.
