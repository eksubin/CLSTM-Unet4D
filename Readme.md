# Project Directory Overview
This project implements a 3D U-Net model enhanced with Convolutional Long Short-Term Memory (CLSTM) cells. This architecture enables the U-Net to effectively leverage temporal resolution by processing both spatial and temporal dimensions in volumetric data. The integration of CLSTM cells allows the model to capture complex dependencies across time, making it particularly suitable for analyzing 4D medical imaging datasets such as dynamic MRI or CT scans, where both spatial and temporal information are crucial for accurate interpretation.

## Files in this Folder
- `clstm_unet3D_time.py`: This file contains the implementation for processing a stack of 5 3D volumes using the CLSTM UNet3D model for segmentation tasks. It utilizes the model to effectively analyze the temporal dynamics within the volumetric data, facilitating enhanced segmentation accuracy across the multiple frames.
- `mock_training_clstm_unet.py`: This code is designed to perform a test run of the CLSTM UNet3D model using mock data generated with MONAI, a library for medical imaging. The script simulates a training loop to demonstrate the functionality of the model. It creates synthetic 4D data and employs a simple training setup for validation purposes.

- `model.py`: This file defines the neural network architecture, including the CLSTM UNet3D model.
 of epochs.
- `requirements.txt`: Lists the required Python libraries and their versions needed to run the project.
- `utils.py`: A utility file with additional helper functions used throughout the project.

## CLSTM UNet3D

The CLSTM UNet3D model is a convolutional neural network architecture that combines convolutional and LSTM (Long Short-Term Memory) layers for the purpose of processing volumetric data. It is particularly useful for tasks such as semantic segmentation in medical imaging. 

### Key Features

- **Convolutional Layers**: Capture spatial features from 3D inputs.
- **LSTM Layers**: Integrate temporal information, allowing the model to learn dependencies over time sequences.
- **Skip Connections**: Enable better gradient flow and help in preserving spatial features across layers.

### Applications

The CLSTM UNet3D architecture can be applied in various fields, including:

- Medical image analysis (e.g., MRI, CT scans)
- 3D object detection and segmentation
