# DL_Assignment2

## Student Name - Mith Concio

This repository is to serve as my submission for assignment 2 in the deep learning elective.

## About the Object Detection Model:
The model is high resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone, this model was chosen because it's relatively lightweight compared to the other available models and without compromising much performance, it's lightweightedness also made it easier to run for real-time object detection, which is critical for the video demo.

## Notes and Assumptions:
The assumption is that libraries that were used in the pytorch tutorial are all already installed (cython, pycocotools, torch, torchvision, etc.), and other libraries used to download the dataset (gdown) will be installed by train.py and test.py when they are run. In case any of the modules are missing, the command pip install -r requirements.txt can be used to install the missing modules. 

## Usage:
Assuming the whole git repository is cloned in one directory, the only things that have to be run are either train.py or test.py, all the other files are only there so that these two can run properly.




