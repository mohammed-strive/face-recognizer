#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:55:17 2020

@author: jvidyad
"""
from shutil import copy, rmtree
import argparse

import cv2
import os

from get_faces_from_image import get_faces_from_image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--target-image", required=True,
	help="path to target image")
args = vars(ap.parse_args())

face_dir = 'Faces'

if os.path.exists(face_dir):
    rmtree(face_dir)
    
os.mkdir(face_dir)

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

target_image = args['target_image']

conf_threshold = args["confidence"]

image = cv2.imread(target_image)

faces_detected = get_faces_from_image(image, detector, 
                                      conf_threshold)

if faces_detected is not None:
    for ii,fc in enumerate(faces_detected):
        file_name = 'image_{:d}.jpg'.format(ii)
        cv2.imwrite(os.path.join(face_dir, file_name), fc)