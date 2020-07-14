#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:47:56 2020

@author: jvidyad
"""
import os

from pprint import pprint

from shutil import rmtree

import cv2

from imutils import paths

import argparse

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from keras_vggface.vggface import VGGFace

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
ap.add_argument("-g", "--cluster", default='Clusters',
	help="path to OpenCV's deep learning face detector")

args = vars(ap.parse_args())

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

print("Images : {}".format(imagePaths))

clusters = []

for ii,image_path in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(ii + 1,
		len(imagePaths)))
    
    embed_vecs, boxes, image = create_embeddings(image_path, detector,
                                                 embedder,
                                                 args["confidence"])
    
    if embed_vecs is None:
        continue
    #print(type(embed_vecs))
    assert type(embed_vecs)==np.ndarray
    assert embed_vecs.ndim==2
    #print(embed_vecs.shape)
    #import sys
    #sys.exit(0)
    
    if len(clusters)==0:
        for kk in range(len(embed_vecs)):
            clusters.append({'images':[image], 'vectors':
                np.expand_dims(embed_vecs[kk], 0), 'boxes':[boxes[kk]], 
                'image_name': [image_path]})
        
    else:
        for kk in range(len(embed_vecs)):
            for jj in range(len(clusters)):
                cluster_centroid = np.mean(clusters[jj]['vectors'], 0)
                sim = cosine_similarity(np.expand_dims(embed_vecs[kk], 0), 
                                        np.expand_dims(cluster_centroid, 0))
                
                
                sim = np.squeeze(sim)
                
                assert sim.ndim==0
                
                if sim>=0.7:
                    clusters[jj]['images'].append(image)
                    clusters[jj]['vectors'] = np.vstack((
                            clusters[jj]['vectors'], embed_vecs[kk]))
                    clusters[jj]['boxes'].append(boxes[kk])
                    clusters[jj]['image_name'].append(image_path)
                    break
                
                if jj==len(clusters)-1:
                    clusters.append({'images':[image], 'vectors':
                                    np.expand_dims(embed_vecs[kk], 0), 
                                    'boxes':[boxes[kk]], 
                                    'image_name': [image_path]})

pprint([x['image_name'] for x in clusters])

clusters_dir = args['cluster']

if os.path.exists(clusters_dir):
    rmtree(clusters_dir)
    
os.mkdir(clusters_dir)

for ii, clust in enumerate(clusters):
    clust_dir = os.path.join(clusters_dir, str(ii+1))
    os.mkdir(clust_dir)
    
    images = clust['images']
    
    for kk in range(len(images)):
        #copy(img, clust_dir)
        #im = cv2.imread(img)
        im = images[kk].copy()
        (startX, startY), (endX, endY) = clust['boxes'][kk]
        im = cv2.rectangle(im, (startX, startY), (endX, endY), 
                           (255, 0, 0), 2)
        img_name = clust['image_name'][kk].split(os.sep)[-1]
        cv2.imwrite(os.path.join(clust_dir, img_name), im)
