import argparse
import logging
import os
import pickle
import sys
import time
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from tensorflow.python.platform import gfile
from embeddings import (
    filter_dataset, split_dataset, get_dataset
)
import embeddings

logger = logging.getLogger(__name__)

def main(input_directory, model_path, classifier_output_path, batch_size,
    num_threads, num_epochs, min_images_per_labels, split_ratio, is_train=True):
    """
    Loads images from :param input_dir, creates embeddings using a model defined at :param model_path, and trains
     a classifier outputted to :param output_path
     
    :param input_directory: Path to directory containing pre-processed images
    :param model_path: Path to protobuf graph file for facenet model
    :param classifier_output_path: Path to write pickled classifier
    :param batch_size: Batch size to create embeddings
    :param num_threads: Number of threads to utilize for queuing
    :param num_epochs: Number of epochs for each image
    :param min_images_per_labels: Minimum number of images per class
    :param split_ratio: Ratio to split train/test dataset
    :param is_train: bool denoting if training or evaluate
    """
    start = time.time()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        train_set, test_set = _get_test_and_train_set(input_directory,
            min_num_images_perl_label=min_images_per_labels, split_ration=split_ratio)
