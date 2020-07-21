#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:56:10 2020

@author: jvidyad
This script is used to get the cropped faces from an image
"""
import numpy as np

import cv2

import imutils

def get_faces_from_image(image, detector, conf_threshold):
    ''' Given an opencv image object, the detector and confidence
    threshold, this function returns the faces in the image.
    '''
    image = imutils.resize(image, width=600)
    
    (h, w) = image.shape[:2]
    
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    faces = []
    
    if len(detections) > 0:
        
        for i in range(len(detections[0, 0, :, 2])):
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
        #i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
    		# ensure that the detection with the largest probability also
    		# means our minimum probability test (thus helping filter out
    		# weak detections)
            if confidence > conf_threshold:
    			# compute the (x, y)-coordinates of the bounding box for
    			# the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                if startX<0 or startY<0 or endX>w or endY>h:
                    continue
    			# extract the face ROI and grab the ROI dimensions
                
                temp = image.copy()
                
                face = temp[startY:endY, startX:endX]
                
                faces.append(face)
                
        if len(faces)>0:
            return faces
        else:
            return None
        
    else:
        return None