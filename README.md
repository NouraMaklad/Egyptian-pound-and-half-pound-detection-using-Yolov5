# Egyptian Pound Detection Model

## Overview
This repository contains a custom object detection model for detecting Egyptian pound and half-pound notes using YOLOv5. It includes programs for counting money and measuring bottle dimensions.

## Components
- **Dataset:** Collection of images of Egyptian pounds and half-pounds in various sizes and positions.
- **Programs:**
  - **Money Counter:** Counts the total amount of money in an image and annotates the image with the total.
  - **Size with a Pound:** Measures the width and height of a bottle using a pound note as a reference and annotates the image with the dimensions.
- **Model:** The trained YOLOv5 model saved as `last.pt`.

## Installation
1. Clone the repository:
   git clone [(https://github.com/NouraMaklad/Egyptian-pound-and-half-pound-detection-using-Yolov5.git)]
2. Navigate to the project directory:
   cd Egyptian pound and half-pound detection using Yolov5
3. Install required packages:
   pip install -r requirements.txt
## Usage
1. **Money Counter:**
 - Input: Image of pounds and half-pounds.
 - Command to run:
      python money_counter.py [input-image]
 - Output: Annotated image with the total amount.
2. **Size with a Pound:**
 - Input: Image of a pound next to a bottle.
 - Command to run:
      python size_with_pound.py [input-image]
 - Output: Annotated image with the bottle dimensions.
## Performance
The model achieves an accuracy of 95% in detecting currency notes and processes images in under 2 seconds.
