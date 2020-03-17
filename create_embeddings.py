#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:55:18 2020

@author: jvidyad
"""
import os
import cv2

import imutils

import numpy as np

def create_embeddings(imagePath, detector, embedder, conf_threshold):
    #name = imagePath.split(os.path.sep)[-2]
	# load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(image.shape)
    (h, w) = image.shape[:2]

	# construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    vecs = np.reshape(np.array([]), (0, 128))
    img = image
    
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
    			# extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                
                img = cv2.rectangle(img, (startX, startY), (endX, endY), 
                                    (255,0,0), 1)
                
    			# ensure the face width and height are sufficiently large
                #if fW < 20 or fH < 20:
                    #continue
        
        # construct a blob for the face ROI, then pass the blob
    			# through our face embedding model to obtain the 128-d
    			# quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
    				(96, 96), (0, 0, 0), swapRB=True, crop=False)
                print(type(faceBlob))
                print(faceBlob.shape)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                
                vecs = np.vstack((vecs, vec))
                
        file_name = imagePath.split(os.sep)[-1]
        cv2.imwrite(os.path.join('/tmp', file_name), img)        
        assert vecs.ndim==2        
        return vecs if len(vecs)>0 else None
    
    else:
        return None