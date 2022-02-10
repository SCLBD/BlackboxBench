"""Downloads a model, computes its SHA256 hash and unzips it
   at the proper location."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import sys
import tarfile
import zipfile

from utils.misc import data_path_join


def dm(urls, data_dir):
    for url in urls:
        fname = data_path_join(data_dir, url.split('/')[-1].split('?')[0])  # get the name of the file

        # model download
        if not os.path.exists(fname):
            print('Downloading models')
            if sys.version_info >= (3,):
                import urllib.request
                urllib.request.urlretrieve(url, fname)
            else:
                import urllib
                urllib.urlretrieve(url, fname)

        # computing model hash
        sha256 = hashlib.sha256()
        with open(fname, 'rb') as f:
            data = f.read()
            sha256.update(data)
        print('SHA256 hash: {}'.format(sha256.hexdigest()))

        # extracting model
        print('Extracting model')
        if fname.endswith('.tar.gz'):
            opener = tarfile.open(fname, 'r:gz')
        else:
            opener = zipfile.ZipFile(fname, 'r')

        with opener as model_zip:
            model_zip.extractall(data_path_join(data_dir))
            print('Extracted model in {}'.format(data_path_join(data_dir)))


# cifar models
data_dir = 'cifar10_models'
os.makedirs(data_path_join(data_dir), exist_ok=True)
urls = [
    'https://www.dropbox.com/s/cgzd5odqoojvxzk/natural.zip?dl=1',
    'https://www.dropbox.com/s/g4b6ntrp8zrudbz/adv_trained.zip?dl=1',
    'https://www.dropbox.com/s/ywc0hg8lr5ba8zd/secret.zip?dl=1'
]
dm(urls, data_dir)

# mnist models
data_dir = 'mnist_models'
os.makedirs(data_path_join(data_dir), exist_ok=True)
urls = [
    'https://github.com/MadryLab/mnist_challenge_models/raw/master/natural.zip',
    'https://github.com/MadryLab/mnist_challenge_models/raw/master/secret.zip',
    'https://github.com/MadryLab/mnist_challenge_models/raw/master/adv_trained.zip'
]
dm(urls, data_dir)

# imagenet models
data_dir = 'imagenet_models'
os.makedirs(data_path_join(data_dir), exist_ok=True)
urls = ['http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz']
dm(urls, data_dir)
