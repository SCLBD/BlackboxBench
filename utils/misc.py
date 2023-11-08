"""
Helper functions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

DIR = 'B-Box'

def get_dataset_shape(dset_name):
    if dset_name == 'mnist':
        dset_shape = (784,)
    elif dset_name == 'cifar10':
        dset_shape = (32, 32, 3)
    elif dset_name == 'imagenet':
        dset_shape = (299, 299, 3)
    elif dset_name == 'image_sub':
        dset_shape = (299, 299, 3)
    else:
        raise Exception('Unsupported dataset for attack yet')

    return dset_shape

def create_dir(_dir):
    """Create a directory, skip if it exists"""
    os.makedirs(_dir, exist_ok=True)


def get_src_dir():
    """returns the {REPO_PATH}/src/"""
    _dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(_dir) # _dir[:_dir.rfind(DIR) + len(DIR)]


def get_config_dir():
    """returs the {REPO_PATH}/d"""
    return os.path.join(get_src_dir(), 'config-jsons')


def get_data_dir():
    """returs the {REPO_PATH}/data"""
    return os.path.join(get_src_dir(), 'data')


def src_path_join(*kwargs):
    """
    reutrns path to the file whose dir information are provided in kwargs
    similar to `os.path.join`
    :param kwargs:
    :return:
    """
    return os.path.join(get_src_dir(), *kwargs)


def data_path_join(*kwargs):
    """
    reutrns path to the file whose dir information are provided in kwargs
    similar to `os.path.join`
    :param kwargs:
    :return:
    """
    return os.path.join(get_data_dir(), *kwargs)


def config_path_join(filename):
    """
    returns abs pathname to the config of name `filename`
    assuming it is in `config-jsons`
    :param filename:
    :return:
    """
    return src_path_join('config-jsons', filename)