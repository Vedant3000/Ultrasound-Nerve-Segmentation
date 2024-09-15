# Ultrasound Nerve Segmentation using Machine Learning (U-Net)

## Overview
This repository contains a cutting-edge machine learning project focused on Ultrasound Nerve Segmentation. The goal is to accurately identify and segment nerve structures in ultrasound images, a crucial task in medical imaging for procedures like nerve blocks or surgery. We leverage the powerful U-Net architecture for semantic segmentation and explore the use of Generative Adversarial Networks (GANs) to further enhance the quality of segmentation results.
Medical image segmentation is a highly challenging task due to the variability in imaging, complex structures, and noisy environments. This project demonstrates how deep learning models, particularly U-Net and GANs, can overcome these challenges and provide highly accurate segmentation.

## Project Highlights
### U-Net Architecture:

U-Net is a state-of-the-art convolutional neural network (CNN) specifically designed for biomedical image segmentation.
The model excels in capturing fine details and accurately segmenting nerve structures in ultrasound images.
### Generative Adversarial Networks (GANs):

GANs are employed to enhance the segmentation results by generating higher quality predictions and refining boundaries.
The GAN-based approach helps the model generalize better in cases of noisy or low-quality images.
## Data Augmentation & Preprocessing:

Extensive data preprocessing techniques like resizing, normalization, and augmentation (flipping, rotation, etc.) are applied to ensure robustness and better generalization of the model.
Preprocessed ultrasound images are fed into the model to improve accuracy and reduce overfitting.

## Evaluation Metrics:
Metrics like Dice coefficient, Intersection over Union (IoU), and binary accuracy are used to evaluate the model’s performance.
Visualization techniques are applied to display the segmentation results and compare them with ground truth images.
Key Features
High Accuracy Segmentation: Achieve precise nerve segmentation with U-Net and enhanced segmentation boundaries using GANs.
Data Augmentation: Improve model robustness through effective data augmentation techniques.
Advanced Visualization: Visualize the segmented images and compare the results using intuitive heatmaps and comparison plots.

## Dataset
The dataset used in this project consists of ultrasound nerve images with annotated masks. Each image is paired with a corresponding ground truth mask that marks the location of the nerve structures.

## Requirements
To run this project, you need to install the following dependencies:
```
keras==2.12.0
tensorflow==2.12.0
segmentation_models
tensorflow-io==0.31.0
pandas
matplotlib
seaborn
opencv-python
sklearn
scikit-learn'
```

## Model Architecture
- U-Net:
Encoder-decoder architecture designed for medical image segmentation.
Captures both high-level semantic information and low-level image details, making it ideal for segmentation tasks.
- GANs:
GANs consist of a generator and discriminator that work in tandem, with the generator refining the predicted segmentations and the discriminator ensuring realistic results.

## Results
U-Net achieves highly accurate nerve segmentation, producing precise and smooth boundaries.
GAN-enhanced Segmentation further refines the predictions, especially in challenging scenarios where the nerve structures are less visible or noisy.
