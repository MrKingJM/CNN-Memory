r"""Data loading and other utilities

Use this file to first copy over and pre-process the Omniglot dataset.
Simply call
    python3 data_utils.py

"""

import logging
import os
import tarfile
import pickle
import numpy as np
import wget
from skimage.transform import resize
from skimage.transform import rotate

import tensorflow as tf

MAIN_DIR = ''
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATASET_NAME = DATA_URL.split('/')[-1]
DATA_DIR = '../data'
DATA_ARCHIVE = os.path.join(DATA_DIR, DATASET_NAME)
DATA_UNARCHIVE_DIR = os.path.join(DATA_DIR, DATASET_NAME.split('.')[0])
TRAIN_DIR = os.path.join(DATA_UNARCHIVE_DIR, 'images_background')
TEST_DIR = os.path.join(DATA_UNARCHIVE_DIR, 'images_evaluation')
DATA_FILE_FORMAT = '%s_dataset.pkl'

TRAIN_ROTATIONS = False # augment training data with rotations
TEST_ROTATIONS = False
IMAGE_ORIGINAL_SIZE = 32
IMAGE_NEW_SIZE = 28


def unpickle(filename):
    with open(filename, 'rb') as f:
        dicts = pickle.load(f, encoding='bytes')
    return dicts

def maybe_download_and_extract():
    """Download dataset repo if it does not exists."""
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    if os.path.exists(DATA_ARCHIVE):
        logging.info('It appears that repo already exists.')
    else:
        logging.info('download file %s with urllib.request' % (DATASET_NAME))

        wget.download(DATA_URL, DATA_ARCHIVE)
        #subprocess.check_output('wget %s -O %s' % (DATA_URL, DATA_ARCHIVE), shell=True)


    if os.path.exists(DATA_UNARCHIVE_DIR):
        logging.info('It appears that train & test data has already been')
    else:
        logging.info('It appears that train & test data has not been ungzip and untar')

        os.mkdir(DATA_UNARCHIVE_DIR)

        logging.info('tar now')

        #subprocess.check_output('tar -zxvf %s -C %s --strip-components 1' % (DATA_ARCHIVE, DATA_UNARCHIVE_DIR), shell=True)
        tarfile.open(DATA_ARCHIVE).extractall(DATA_DIR)
        os.rename(os.path.join(DATA_ARCHIVE, 'cifar-10-batches-py'), DATA_UNARCHIVE_DIR)


def get_data():
    """Get data in form suitable for eposodic training

    Returns:
        Train and test data as disctionaries mapping
        label to list examples.
    """
    train_dataset_path = os.path.join(TRAIN_DIR, DATA_FILE_FORMAT % 'train')
    with open(train_dataset_path, 'rb') as f:
        processed_train_data = pickle.load(f, encoding='bytes')

    test_dataset_path = os.path.join(TEST_DIR, DATA_FILE_FORMAT % 'test')
    with open(test_dataset_path, 'rb') as f:
        processed_test_data = pickle.load(f, encoding='bytes')

    train_data = {}
    test_data = {}

    for data, processed_data in zip([train_data, test_data],
                                    [processed_train_data, processed_test_data]):
        for image, label in zip(processed_data['images'],
                                processed_data['labels']):
            if label not in data:
                data[label] = []
            data[label].append(image.reshape([-1]).astype('float32'))

    return train_data, test_data


def convert_image_array_to_RGBs(array):
    length_of_a_channel = IMAGE_ORIGINAL_SIZE ** 2
    iwidth = IMAGE_ORIGINAL_SIZE
    iheight = IMAGE_ORIGINAL_SIZE

    Rs = array[0: length_of_a_channel].reshape(iwidth, iheight)
    Gs = array[length_of_a_channel: length_of_a_channel*2].reshape(iwidth, iheight)
    Bs = array[length_of_a_channel*2: length_of_a_channel*3].reshape(iwidth, iheight)
    img_arr = np.dstack((Rs, Gs, Bs))

    return img_arr

def convert_RGBs_to_Grey(img, width, height):
    arr = np.zeros([width, height], dtype=np.uint32)
    for i in range(width):
        for j in range(height):
            arr[i][j] = int(0.299 * img[i][j][0] + 0.587 * img[i][j][1] + 0.114 * img[i][j][2])
            if arr[i][j] > 255:
                arr[i][j] = 255
            elif arr[i][j] < 0:
                arr[i][j] = 0
    return arr


def crawl_directory(augment_with_rotations=False,
                    train_or_not=True):
    """Crawls data directory and returns stuff."""
    images = []
    labels = []
    info = []
    full_filenames = []

    if train_or_not:
        for i in range(1, 6):
            filename = 'data_batch_%s' % i
            filename = os.path.join(DATA_UNARCHIVE_DIR, filename)
            full_filenames.append(filename)
    else:
        filename = 'test_batch'
        filename = os.path.join(DATA_UNARCHIVE_DIR, filename)
        full_filenames.append(filename)

    for the_file in full_filenames:
        logging.info('Reading files from %s' % the_file)
        dicts = unpickle(the_file)
        for image in zip(dicts[b'data'], dicts[b'labels'], dicts[b'filenames']):
            data = image[0]
            label = image[1]
            filename = image[2]

            img = convert_image_array_to_RGBs(data)
            img = convert_RGBs_to_Grey(img, IMAGE_ORIGINAL_SIZE, IMAGE_ORIGINAL_SIZE)

            for i, angle in enumerate([0, 90, 180, 270]):
                if not augment_with_rotations  and i > 0:
                    break

                rotate(img, angle)
                images.append(rotate(img, angle))
                labels.append(label)
                info.append(filename)

    return images, labels, info


def resize_images(images, new_width, new_height):
    """Resize images to new dimensions."""
    resized_images = np.zeros([images.shape[0], new_width, new_height], dtype=np.float32)

    for i in range(images.shape[0]):
        resized_images[i, :, :] = resize(images[i,:,:],
                                        (new_width, new_height))
    return resized_images

def write_datafiles(directory, write_file,
                    resize=True, rotate=False,
                    new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE,
                    train_or_not=True):
    """Load and preprocess images from a directory and write them to a file.

    Args:
        directory: Directory of original images
        write_file: Filename to write to.
        resize: Whether to resize the images
        rotate: Whether to augment the dataset with rotations.
        new_width: New resize width
        new_height: New reisze height
        first_label: Label to start with.

    Returns:
        Numbers of new labels created.
    """

    imgwidth = IMAGE_ORIGINAL_SIZE
    imgheight = IMAGE_ORIGINAL_SIZE

    logging.info('Reading the data.')
    images, labels, info = crawl_directory(augment_with_rotations=rotate,
                                           train_or_not=train_or_not)

    images_np = np.zeros([len(images), imgwidth, imgheight], dtype=np.bool)
    labels_np = np.zeros([len(labels)], dtype=np.uint32)

    for i in range(len(images)):
        images_np[i, : , : ] = images[i]
        labels_np[i] = labels[i]

    if resize:
        logging.info('Resizing images.')
        resized_images = resize_images(images_np, new_width, new_height)

        logging.info('Writing resized data in float32 format.')
        data = {'images': resized_images,
                'labels': labels_np,
                'info': info}
        write_file = os.path.join(directory ,write_file)
        with open(write_file, 'wb') as f:
            pickle.dump(data, f);
    else:
        logging.info('Writing original sized data in boolean format')
        data = {'images': resized_images,
                'labels': labels_np,
                'info': info}
        write_file = os.path.join(directory ,write_file)
        with open(write_file, 'wb') as f:
            pickle.dump(data, f);

    #return len(np.unique(labels_np))

def preprocess_dataset():
    """Download and preprocess raw dataset

    Downloads the data from web
    Then load the images, aument with rotations if desired.
    Resize the images and write them to a pickle file.
    """

    maybe_download_and_extract()

    if not os.path.exists(TRAIN_DIR):
        os.mkdir(TRAIN_DIR)

    directory = TRAIN_DIR
    write_file = DATA_FILE_FORMAT % 'train'
    write_datafiles(
        directory, write_file, resize=True, rotate=TRAIN_ROTATIONS,
        new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE, train_or_not=True)


    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)

    directory = TEST_DIR
    write_file = DATA_FILE_FORMAT % 'test'
    write_datafiles(
        directory, write_file, resize=True, rotate=TEST_ROTATIONS,
        new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE,
         train_or_not=False)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    preprocess_dataset()
