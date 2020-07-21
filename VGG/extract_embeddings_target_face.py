#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 20:23:31 2020

@author: jvidyad
This script gets images which contain a specific target face. The target 
face has already been extracted and saved in a file.
"""

from shutil import copy, rmtree
from imutils import paths
import numpy as np
import argparse
#import imutils
#import pickle
import cv2
import os

from sklearn.metrics.pairwise import cosine_similarity

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

from create_embeddings import create_embeddings

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
#ap.add_argument("-e", "--embeddings", required=True,
#	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
#ap.add_argument("-m", "--embedding-model", required=True,
#	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--target-image", required=True,
	help="path to target image")
args = vars(ap.parse_args())

detected_images_dir = 'Detected'

if os.path.exists(detected_images_dir):
    rmtree(detected_images_dir)
    
os.mkdir(detected_images_dir)

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
#embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
embedder = VGGFace(model='resnet50', include_top=False, 
                   input_shape=(224, 224, 3), pooling='avg')
# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

target_image = args['target_image']

conf_threshold = args["confidence"]

target_face = cv2.imread(target_image)

faceBlob = cv2.dnn.blobFromImage(target_face, 1.0,
    				(224, 224), (0, 0, 0), swapRB=True, crop=False)
faceBlob = np.rollaxis(faceBlob, 2, 1)
faceBlob = np.rollaxis(faceBlob, 3, 2)
faceBlob = preprocess_input(faceBlob, version=2)

target_vec = np.squeeze(embedder.predict(faceBlob))

#print(target_vec)
#print(target_vec)
assert target_vec.ndim==1

not_detected = 'NotDetected'

if os.path.exists(not_detected):
    rmtree(not_detected)
    
os.mkdir(not_detected)

images_present = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
    print(imagePath)
    #name = imagePath.split(os.path.sep)[-2]
	# load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
    embed_vecs, boxes, image = create_embeddings(imagePath, detector, embedder,
                                   conf_threshold)
    
    if embed_vecs is None:
        copy(imagePath, not_detected)
        continue
    
    #print(embed_vecs.shape)
    sims = cosine_similarity(embed_vecs, np.expand_dims(target_vec, 0))
    #print(sims)
    sims = np.ravel(sims)
    #print(sims)
    
#    if type(sims)==float:
#        sims = np.array([sims])
        
    #print(sims.shape)
    assert sims.ndim==1
    
    max_sim = np.argmax(sims)
    
    if sims[max_sim]>=0.7:
        #total += 1
        (startX, startY), (endX, endY) = boxes[max_sim]
        images_present.append(imagePath)
        image = cv2.rectangle(image, (startX, startY), (endX, endY), 
                              (255, 0, 0), 2)
        file_name = imagePath.split(os.sep)[-1]
        cv2.imwrite(os.path.join(detected_images_dir, file_name), image)
            
print("Images in which target is present:")

for image in images_present:
    print(image)