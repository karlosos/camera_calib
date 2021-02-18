import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

DATASET_JPEGIMAGES_PATH = Path("C:\\Users\\karol\\Desktop\\MastersThesis\\calib_dataset\\out\\JPEGImages")
DATASET_ANNOTATIONS_PATH = Path("C:\\Users\\karol\\Desktop\\MastersThesis\\calib_dataset\\out\\Annotations")

OUTPUT_PATH = Path("C:\\Users\\karol\\Desktop\\MastersThesis\\calib_dataset\\out\\dataset_02")
OUTPUT_TRAIN_PATH = OUTPUT_PATH.joinpath('train')
OUTPUT_TEST_PATH = OUTPUT_PATH.joinpath('test')

def prepare_dataset():
    # Split filenames
    file_names = os.listdir(DATASET_JPEGIMAGES_PATH)
    file_names = [Path(file).with_suffix('') for file in file_names]
    X_train, X_test = train_test_split(
        file_names, test_size=50, random_state=1337
    )

    # Create output folders
    if not os.path.exists(OUTPUT_TRAIN_PATH.joinpath('JPEGImages')):
        os.makedirs(OUTPUT_TRAIN_PATH.joinpath('JPEGImages'))
    if not os.path.exists(OUTPUT_TEST_PATH.joinpath('JPEGImages')):
        os.makedirs(OUTPUT_TEST_PATH.joinpath('JPEGImages'))

    if not os.path.exists(OUTPUT_TRAIN_PATH.joinpath('Annotations')):
        os.makedirs(OUTPUT_TRAIN_PATH.joinpath('Annotations'))
    if not os.path.exists(OUTPUT_TEST_PATH.joinpath('Annotations')):
        os.makedirs(OUTPUT_TEST_PATH.joinpath('Annotations'))

    # Copy files 
    for file_path in X_train:
        jpeg_filename = file_path.with_suffix('.jpg')
        xml_filename = file_path.with_suffix('.xml')
        shutil.copyfile(DATASET_JPEGIMAGES_PATH.joinpath(jpeg_filename), OUTPUT_TRAIN_PATH.joinpath('JPEGImages').joinpath(jpeg_filename))
        shutil.copyfile(DATASET_ANNOTATIONS_PATH.joinpath(xml_filename), OUTPUT_TRAIN_PATH.joinpath('Annotations').joinpath(xml_filename))

    for file_path in X_test:
        jpeg_filename = file_path.with_suffix('.jpg')
        xml_filename = file_path.with_suffix('.xml')
        shutil.copyfile(DATASET_JPEGIMAGES_PATH.joinpath(jpeg_filename), OUTPUT_TEST_PATH.joinpath('JPEGImages').joinpath(jpeg_filename))
        shutil.copyfile(DATASET_ANNOTATIONS_PATH.joinpath(xml_filename), OUTPUT_TEST_PATH.joinpath('Annotations').joinpath(xml_filename))


if __name__ == "__main__":
    prepare_dataset()
