# Face Surveillance and Following ROS Program

## Overview

### This program consists of two modes:

#### Face Recognition Surveillance

* Get frames from camera. 
* Detect the face(s) using either HOG or CNN algorithms. 
* Put bounding boxes around the face(s).
* Use deep metric learning algorithm to recognize the face(s) according to faces dataset.
* Once new face is recognized (not recognized before by the program)
    - Frame is stored.
    - Name(s), Datetime, Image Path are stored in CSV file.

#### Face Recognition Following

* First it do the same as Face Recognition Surveillance program. 
* Once it sees face, it rotates towards it until the face bounding box become in the center robot’s camera.
* Move linearly until person is reached. 

## Video
[![Demo](gif_image.gif)](https://www.youtube.com/watch?v=mvrjb8ecock&t=17s)

## Authors 

* Abdulrhman Saad
* Mohanned Ahmed 
