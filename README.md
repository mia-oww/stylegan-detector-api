StyleGAN Detector Revamp (Work in progress)

Overview:

This project documents the refactorization of a legacy StyleGAN image detection system that was initially developed to determine whether a given image of a human face was generated using StyleGAN technology or a real image. The core detection mechanism relies on neural networks implemented in Python, which analyze facial images to effectively differentiate between AI-generated and authentic human faces. Currently, the original source code is undergoing a careful refactoring process to transform it into a modular and scalable API architecture that can support future expansion and integration. At this time, the neural networks powering the detection system are actively being trained and optimized to improve their accuracy and reliability. This repository is dedicated exclusively to the development of the API layer, while the web interface responsible for user interaction and experience is being developed in a separate repository. Once the API is fully operational and the models are sufficiently trained, the frontend web interface will be integrated to provide a seamless and efficient user-facing platform for AI image detection.

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

