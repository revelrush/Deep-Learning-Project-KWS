#Keyword Search (KWS) Application

## Student Name - Mith Concio

This repository contains the codebase for my assignment 3 in the deep learning elective under Dr. Atienza. In this project I implemented and trained a Vision Transformer (ViT) model using the Pytorch Lightning framework with the architecture and dataset provided by the instructor. Moreover, I optimized the configuration of the model’s attention algorithm to improve the model’s accuracy from a baseline of ~63% to 93%, while keeping the model as lightweight as possible. Finally there is also a basic demo application using the model that is able to detect up to 37 different keywords using real-time audio from a microphone.

## About the KWS Model:
The model uses the Vit Transformer architechture provided by the course handler that had the following specifications: 4 heads, a depth of 12 and has an embedding dimension of 64. The model was implemented using the Pytorch Lightning framework, and the dataset used to for training was from the original KWS paper. The highest test accuracy recorded for this model is 93%. The project converts audio to a spectrograph before feeding it to the model, that's how the ViT was used to perform classification on audio.  

## Usage:
### Dependencies
To make sure that none of the required modules are missing, the run the command pip install -r requirements.txt on your Python Environment before running the codebase. 

### Training Script
Running the train.py file will download the dataset and run the training algorithm from scratch. Once finished, it will produce a file named KWS2.ckpt that can be used for inference, all you have to is rename the checkpoint file used in the demo script. 

### Demo Script

All you need to do is to run the kws-infer.py file, it should open a simple GUI that shows which word the model detects like this:

![image](https://github.com/revelrush/Deep-Learning-Project-KWS/assets/84671795/e7a78525-8505-45b2-ab86-a5dd3bad12d6)

There is a video demo attached that demonstrates the model in action.

![video](kws-demo.mp4)

