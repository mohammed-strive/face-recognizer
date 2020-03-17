#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:47:56 2020

@author: jvidyad
"""
import os

from pprint import pprint

import cv2

from imutils import paths

import argparse

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from create_embeddings import create_embeddings

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
#ap.add_argument("-e", "--embeddings", required=True,
#	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

clusters = []

for ii,image_path in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(ii + 1,
		len(imagePaths)))
    
    embed_vecs = create_embeddings(image_path, detector, embedder,
                                   args["confidence"])
    
    if embed_vecs is None:
        continue
    
    if len(clusters)==0:
        for vec in embed_vecs:
            clusters.append({'images':[image_path], 'vectors':
                np.expand_dims(vec, 0)})
        
    else:
        for vec in embed_vecs:
            for jj in range(len(clusters)):
                cluster_centroid = np.mean(clusters[jj]['vectors'], 0)
                sim = cosine_similarity(np.expand_dims(vec, 0), 
                                        np.expand_dims(cluster_centroid, 0))
                
                
                sim = np.squeeze(sim)
                
                assert sim.ndim==0
                
                if sim>=0.8:
                    clusters[jj]['images'].append(image_path)
                    clusters[jj]['vectors'] = np.vstack((
                            clusters[jj]['vectors'], vec))
                    break
                
                if jj==len(clusters)-1:
                    clusters.append({'images':[image_path], 'vectors':
                                    np.expand_dims(vec, 0)})

pprint([x['images'] for x in clusters])