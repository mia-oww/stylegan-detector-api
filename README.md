StyleGAN Detector Revamp (Work in progress)

Overview:

This repository documents the refactoring of a legacy StyleGAN image detection system, which was originally an untrained Python script utilizing a neural network designed to determine whether a given image of a human face was generated using StyleGAN. The core detection mechanism relies on neural networks implemented in Python, which analyze facial images to differentiate between AI-generated and authentic human faces. Currently, the original source code is undergoing a refactoring process to transform it into a scalable API that can support future integration, specifically within the DetectAI Front-end. At this time, the neural networks are actively being trained and optimized to improve their accuracy and reliability. This repository is dedicated exclusively to documenting the development of the API layer, while the web interface responsible for user interaction is being developed in a separate repository. Once the API is fully operational and the models are sufficiently trained, the frontend web interface will be integrated to provide a seamless and efficient user-facing platform for AI facial image detection.

Original untrained neural network: https://github.com/ksmolko/stylegan-detector/tree/main?tab=readme-ov-file

Datasets being used to train models:

Fake Images: https://drive.google.com/drive/folders/1-5HnXJuN1ofCrCSbbVSH3NnP62BZTh4s 

Real Images: https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL


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
 
Due to the heavy files being used to train the models for this API, not all parts will be documented. As of now, this repository will be updated manually.   


Original Project Background:

The original StyleGAN Detector was designed to detect StyleGAN-generated images using co-occurrence matrices and a neural network classifier. The training and testing scripts were dependent on large datasets and a tightly coupled codebase, which limited scalability and maintainability.


All credits for the original training scrips go to ksmolko.
