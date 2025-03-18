🦴 Bone Fracture Classification Using CNN.

A deep learning project that automates the classification of bone fractures in X-ray images using Convolutional Neural Networks (CNNs). This model aims to assist healthcare professionals in resource-limited settings by providing fast and accurate fracture detection.

.
📌 Project Overview
This project uses a CNN model to classify X-ray images as either fractured or non-fractured. It leverages a dataset of over 10,000+ radiographs from different anatomical regions such as:

Upper & Lower limbs
Lumbar
Hips
Knees


About the Dataset
Publicly available dataset sourced from Kaggle: Bone Fracture Dataset
Classes: Fractured (0) and Not-Fractured (1)
Preprocessing: Grayscale conversion, resizing, normalization

Steps involved before training the data

Resizing:
The grayscale X-ray images were resized using opencv to reduce computational intensity and were resized to a fixed dimension (pxdn x pxdn) to ensure uniformity across the dataset. Pixel values were normalized (scaled between 0 and 1) to facilitate faster convergence during model training.

![image](https://github.com/user-attachments/assets/3fca51e0-440a-4f1d-9d48-3cffb70a6c46)

