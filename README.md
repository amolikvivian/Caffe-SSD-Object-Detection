# Caffe-SSD-Object-Detection
Object Detection using Single Shot MultiBox Detector with Caffe MobileNet on OpenCV in Python.

## SSD Framework
Single Shot MultiBox Detectors can be divided into two parts:
 
 - Extracting Features using a base network
 - Using Convolution Filters to make predictions
 
 This implementation makes use of the MobileNet deep learning CNN architecture as the base network. 

## Caffe Framework
Caffe is a deep learning framework developed by the Berkely AI Research and Community Contributors. Caffe [repo](https://github.com/BVLC/caffe). It is a much faster way of training images with over 6 million images per day using an Nvidia 
K-40 GPU

## Run code
`python detectDNN.py -p Caffe/SSD_MobileNet_prototxt -m Caffe/SSD_MobileNet.caffemodel`

## Article
[Medium](https://medium.com/analytics-vidhya/ssd-object-detection-in-real-time-deep-learning-and-caffe-f41e40eea968)
