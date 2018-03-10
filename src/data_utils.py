r"""Data loading and other utilities

Use this file to first copy over and pre-process the Omniglot dataset.
Simply call
    python3 data_utils.py

"""

import cPickle as pickle
import logging
import os
import subprocess

import numpy as np
from scipy.misc import imresize
from scipy.misc import imrotate
from scipy.ndimage import imread
import tensorflow as tf

MAIN_DIR = ''
REPO_LOCATION = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
