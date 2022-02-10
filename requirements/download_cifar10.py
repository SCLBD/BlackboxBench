"""
Download cifar10 data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

from utils.misc import data_path_join

data_dir = data_path_join('cifar10_data')
os.makedirs(data_dir, exist_ok=True)
url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
fname = data_path_join('cifar10_data', 'cifar-10-python.tar.gz')

if not os.path.exists(fname):
    print('Downloading cifar10')
    if sys.version_info >= (3,):
        import urllib.request

        urllib.request.urlretrieve(url, fname)
    else:
        import urllib

        urllib.urlretrieve(url, fname)

# extracting data
tar = tarfile.open(fname, "r:gz")
tar.extractall(data_dir)
tar.close()
print('Extracted model in {}'.format(data_dir))
