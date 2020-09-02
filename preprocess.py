import argparse
import glob
import logging
import multiprocessing as mp
import os
import time
import cv2
from align_dlib import AlignDlib


logger = logging.getLogger(__name__)
align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__),
    'shape_predictor_68_face_landmarks.dat'))

def main(input_dir, output_dir, crop_dim):
    start_time = time.time()

    pool = mp.Pool(processes=os.cpu_count())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_dir in os.listdir(input_dir):
        image_output_dir = os.path.join(output_dir,
            os.path.basename(os.path.basename(image_dir)))
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

    image_paths = glob.glob(os.path.join(input_dir, "**/*.jpg"))
    
    for index, image_path in enumerate(image_paths):
        image_output_dir = os.path.join(output_dir, os.path.basename(
            os.path.dirname(image_path)))
        output_path = os.path.join(image_output_dir, os.path.basename(image_path))
        pool.apply_async(preprocess_image, (image_path, output_path, crop_dim))

    pool.close()
    pool.join()
    logger.info("Completed in {} seconds".format(time.time() - start_time))

def preprocess_image(input_path, output_path, crop_dim):
    """
    Detect face, align and crop :param input_path. Write output to :param output_path
    :param input_path: Path to input image
    :param output_path: Path to write processed image
    :param crop_dim: dimensions to crop image to
    """

    images = _process_image(input_path, crop_dim)
    for index, image in enumerate(images):
        if image is not None:
            logger.debug("Writing processed file: {}".format(output_path))
            dirname = os.path.dirname(output_path)
            basename = os.path.basename(output_path)
            extension = basename[-1:basename.index(".")]
            if extension is None:
                extension = "jpg"
            filename = "{}/{}_{}.{}".format(dirname, basename, index, extension)
            cv2.imwrite(filename, image)
        else:
            logger.warning("Skipping filename: {}".format(output_path))

def _process_image(input_path, crop_dim):
    image = None

    image = _buffer_image(input_path)

    if image is not None:
        aligned_images = _align_image(image, crop_dim)
    else:
        raise IOError("Error buffering image: {}".format(input_path))

    return aligned_images

def _buffer_image(filename):
    logger.debug("Reading image: {}".format(filename))
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def _align_image(image, crop_dim):
    boundingBoxes = align_dlib.getAllFaceBoundingBoxes(image)
    
    aligned_images = []

    for bb in boundingBoxes:
        aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=align_dlib.INNER_EYES_AND_BOTTOM_LIP)
        if aligned is not None:
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            aligned_images.append(aligned)

    return aligned_images
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input_dir', type=str, default="data", action="store", dest="input_dir")
    parser.add_argument('--output_dir', type=str, default="output", action="store", dest="output_dir")
    parser.add_argument('--crop_dim', type=int, default=180, action="store", dest="crop_dim")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.crop_dim)