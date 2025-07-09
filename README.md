StyleGAN Detector Revamp (Currently In Progress)

Overview:

This project is a modernization of a legacy StyleGAN image detection system originally built to classify whether an image of a human face was generated using StyleGAN or not. I am currently refactoring the original source code into a modular, scalable API and preparing it for integration with a web interface. This is specifically for the API, web-development will be in a seperate Repo.

Link to original project: https://github.com/ksmolko/stylegan-detector/tree/main?tab=readme-ov-file


Technologies:
Python 3.8.6

TensorFlow 2.3.0

Keras 2.4.3

OpenCV 4.1.2

Scikit-Image 0.16.2

NumPy, Matplotlib, Pillow


Current Progress:

 -Refactored monolithic training and testing scripts into modular components
 
 -Set up API framework for future deployment
 
 -Currently retraining models using a prototype dataset of 1,000 PNG images
 
 -Optimizing image processing workflows and co-occurrence matrix generation
 
 -Preparing web integration to enable real-time image authenticity classification
 
 

Original Project Background:

The original StyleGAN Detector was designed to detect StyleGAN-generated images using co-occurrence matrices and a neural network classifier. The training and testing scripts were dependent on large datasets and a tightly coupled codebase, which limited scalability and maintainability.

