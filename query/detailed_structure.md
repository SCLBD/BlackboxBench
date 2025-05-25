
B-box/
  |--attacks/
  | |--decision-based attacks/ # contain the base attack class file and 8 decision-based attack methods
  | |--score-based attacks/ # contain the base attack class file and 7 score-based attack methods
  |--config-jsons/ #  contain json files which set the experiment configuration. You can write json files to configurate your experiment by following our existing format.
  |--datasets/
  | |--cifar10.py # Ultilities for importing CIFAR10 dataset.
  | |--dataset.py # A wrapper for datasets such as MNIST, CIFAR10, ImageNet.
  | |--imagenet.py #  A wrapper for ImageNet validation set, which is a simple loader with the appropriate transforms.
  |--models/ # contain different models for attack.
  |--pics/ # Records of some experiment outputs.
  |--requirments.txt 
  |--utils/ 
  | |--compute.py # implements handy numerical computational functions.
  | |--misc.py # helper functions
  | |--model_loader.py # load different model according to configuration file.
  |--.gitignore
  |--README.md
  |--attack_cifar10.py # main python file to run attacks on CIFAR10.
  |--attack_imagenet.py # main python file to run attacks on ImageNet.
