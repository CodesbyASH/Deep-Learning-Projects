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

Label Encoding:
The dataset was labeled into binary classes:

0 for Fractured
1 for Not-Fractured
This simplified the classification problem into a binary task suited for a sigmoid output layer.


![image](https://github.com/user-attachments/assets/3fca51e0-440a-4f1d-9d48-3cffb70a6c46)

Image after being resized.
![image](https://github.com/user-attachments/assets/af37ded3-8e51-4147-a561-015054381955)


Data Augmentation:
Why was this necessary?
Data augmentation introduces diversity to the training data, enabling the model to generalize better and reducing overfitting risk.

To mitigate overfitting and improve the model’s ability to generalize to unseen images, I applied data augmentation techniques:

Random Rotation
Simulates images captured at different angles.

Horizontal/Vertical Flipping
Mimics real-world variability in patient positioning.

Zooming & Shifting
Helps the model recognize fractures at varying distances and locations within the image.

Brightness Adjustments
Improves robustness against varying X-ray exposure levels.

![image](https://github.com/user-attachments/assets/d74525b4-23f1-4dfb-abae-40427984d50c)


🏗️The Model Architecture
Conv2D Layer 1: 32 filters (3x3), ReLU activation
Two sets of convolutional layers (32 & 64 filters) were used to extract hierarchical features from X-ray images, such as edges, contours, and fracture patterns.

MaxPooling2D Layer 1: Pooling (2x2)
Applied after each convolutional block to downsample the feature maps, reducing dimensionality and computational load while retaining critical features.

Conv2D Layer 2: 64 filters (3x3), ReLU activation
MaxPooling2D Layer 2: Pooling (2x2)
Flatten Layer
Dense Layer: 128 units, ReLU activation
The extracted features were flattened and passed through fully connected layers to make the final classification.

Output Layer: 1 unit, Sigmoid activation (for binary classification)
Optimizer: Adam
Loss Function: Binary Crossentropy

Regularization: Dropout & Data Augmentation
Introduced dropout between layers to prevent overfitting by randomly deactivating neurons during training.


Model Compilation & Training
Loss Function: Binary Crossentropy
Used due to the binary nature of the classification task.

Optimizer: Adam
Selected for its efficiency and adaptive learning capabilities.

Metrics: Accuracy
Primary metric to track the model’s performance.

Epochs: 10 iterations
Chosen to balance training time with model performance

![image](https://github.com/user-attachments/assets/f4689ca0-58a9-451a-8e3e-3a34ca5211bb)

Evaluation Metrics
Post-training, I evaluated the model using:

Accuracy (achieved 89.7% on test data)
Precision & Recall
Confusion Matrix

The use of these metrics was because they provide a holistic view of model performance, especially important in healthcare applications where minimizing false negatives is critical.


🌐 Web Interface(To demonstrate the exact point of fractured region if there lies a fracture in the X-ray)
A Flask-based web interface was developed to:

Upload X-ray images
Run fracture classification in real-time
Display prediction (fractured / not-fractured)

![image](https://github.com/user-attachments/assets/ada9f64e-55e5-4cf6-a28a-9495bcfec2aa)


Future Improvements and Lessons Learned:
Incorporating transfer learning with pre-trained models (e.g., ResNet, VGG)
Using this model it can be understood that the implementation of CNN for bone fracture classification has a significant potential, providing a fast and accurate method for medical image analysis.

Hands-on experience in managing medical imaging data and real-world problems like image quality variance and dataset imbalance.
A sound knowledge of hyperparameter tuning and CNN architectures.
Crucial usability and interpretability, particularly when developing healthcare solutions.







