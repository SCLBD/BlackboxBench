# B-Box: an adversarial black-box attack methods toolbox to evaluate the robustness of Machine Learning models in Pytorch.

![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.2.0-DodgerBlue.svg?style=plastic)
![CUDA 10.0](https://img.shields.io/badge/cuda-10.0-DodgerBlue.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic)

B-Box is a toolbox containing mainstreamed adversarial black-box attack methods implemented based on PyTorch. You can easily adopt it to evaluate robustness of your ML models or design the better attack methods. Meanwhile, we also provide a benchmark which evaluate their attack performance against several defense methods. Currently, we support:

- Datasets: `CIFAR-10, ImageNet.`
- Attack methods: 
	- query-based attack methods: ` 7 score-based attack and 8 decision-based attack methods.`
	- transfer attack methods: `Coming soon!`
---
<font size=5><center><b> Table of Contents </b> </center></font>

- [Quick Start](#quick-start)
- [Dependency](#dependency)
- 


---

### Quick Start:

After modifying the attacks config files in `config-jsons` as desired, include config files of the considered attacks in `attack_cifar10.py` as follows:

```
python attack_cifar10.py ***.json
```

### Dependency



### Instructions:
The code is well-organized. Users can use and extend upon it with little efforts. 

B-box is organized as follows:
```
B-box/
  |--attacks/
  | |--decision-based attacks/ # contain 9 decision-based attack methods, presented by python files.
  | |--score-based attacks/ # contain 8 score-based attack methods, presented by python files.
  | |--transfer-based attacks/ # contain 11 transfer-based attack methods in flag.py
  |--config-jsons/ #  contain json files which set the experiment configuration. You can write json files to configurate your experiment by following our existing format.
  |--datasets/
  | |--cifar10.py # Ultilities for importing CIFAR10 dataset.
  | |--dataset.py # A wrapper for datasets such as MNIST, CIFAR10, ImageNet.
  | |--imagenet.py #  A wrapper for ImageNet validation set, which is a simple loader with the appropriate transforms.
  |--models/ # contain different models for attack.
  |--pics/ # Records of some experiment outputs.
  |--requirments/ # contain the conda enviroment requirment and pyhon files about how to download CIFAR10 dataset and ImageNet dataset.
  |--utils/ 
  | |--compute.py # implements handy numerical computational functions.
  | |--misc.py # helper functions
  | |--model_loader.py # load different model according to configuration file.
  |--.gitignore
  |--README.md
  |--attack_cifar10.py # main python file to run on CIFAR10.
  |--attack_imagenet.py #main python file to run on ImageNet.
```
Users can modify the configuration file (***.json) to run different attack methods on different models with l-infty norm or l-2 norm.

### 1. Pretrain models:
Here, we test the contained attack methods on the below models.
+ **CIFAR-10**: ResNet-50, WideResNet-28, AT-l_inf-WideResNet-28 [(with extra data (Gowal et al., 2020))](https://arxiv.org/abs/2010.03593), AT-l_inf-WideResNet-28 [(with data from DDPM (Rebuffi et al., 2021))](https://arxiv.org/abs/2103.01946).
For ResNet-50 and WideResNet-28, we train them by using the code from this [github repo](https://github.com/kuangliu/pytorch-cifar). 

+ **ImageNet**: ResNet-50, Inception-v3, AT-l_inf-ResNet-50 (4/255) [(Salman et al., 2020)](https://github.com/microsoft/robust-models-transfer), FastAT-l_inf-ResNet-50 (4/255) [(Wong et al., 2020)](https://github.com/locuslab/fast_adversarial), Feature-Denosing-ResNet-152 [(Xie et al., 2019)](https://github.com/facebookresearch/ImageNet-Adversarial-Training).
For ResNet-50 and Inception-v3, we use the provided pretrained model from torchvision.

### 2. Load pretrained models
Before users run the main file (`attack_cifar10.py` & `attack_imagenet.py`), they need to load pretrained model with `.pth` file. The following part is an example of how to load `Wide-Resnet-28` pretrained on `CIFAR10`. Users need to put pretrained model file '`cifar_wrn_28.pth`' into '`pretrained_models/`' and change the file path accordingly in `model_loader.py`.

```
elif model_name == 'wrn28':
	TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_adv/')
	filename = 'cifar_wrn_28.pth'
	pretrained_model = wrn.WideNet()
	pretrained_model = torch.nn.DataParallel(pretrained_model)
	checkpoint = torch.load(os.path.join(TRAINED_MODEL_PATH, filename))
	# if hasattr(pretrained_model, 'module'):
	#     pretrained_model = pretrained_model.module
	pretrained_model.load_state_dict(checkpoint['net'])
```

### 3. Start to attack via different methods on different pretrained models.

The following part is about how to modify a config-json file as desired. Here is an example config-json file for `Signopt Attack` on `Wide-Resnet-28` (`CIFAR10 `dataset).
```
{
	"_comment1": "===== DATASET CONFIGURATION =====",
	"dset_name": "cifar10", #Users can change the dataset here.
	"dset_config": {},
	"_comment2": "===== EVAL CONFIGURATION =====",
	"num_eval_examples": 10000,  
	"_comment3": "=====ADVERSARIAL EXAMPLES CONFIGURATION=====",
	"attack_name": "SignOPTAttack", #We choose Signopt attack method.
	"attack_config": {
	"batch_size": 1,
	"epsilon": 255,
	"p": "2", #set the perturbation norm to be l-2 norm, while "inf" represents l-infty norm.
	"alpha": 0.2,
	"beta": 0.001,
	"svm": false,
	"momentum": 0,
	"max_queries": 10000, #We use unified maximum queries number to be 10000. 
	"k": 200,
	"sigma": 0
	  },
	"device": "gpu",
	"modeln": "wrn28", #the name should be in accordance with the one in model_loader.py
	"target": false, #Users can choose to run targeted attack(true) or untargeted attack(false).
	"target_type": "median",
	"seed":123
	}
  
```
We set the maxium queries to be `10000` on all tests and the attack budget will be set uniformly by 
```
CIFAR: 	l_inf：0.05 = 12.75/255, l_2: 1 = 255/255
ImageNet: l_inf: 0.05 =  12.75/255, l_2: 5= 1275/255	
```

where `l_inf` represents l-infty norm perturbation and `l_2` represents l-2 norm perturbation.



### 4.Running Attacks

After modifying the attacks config files in `config-jsons` as desired, include config files of the considered attacks in `attack_cifar10.py` as follows:

```
python attack_cifar10.py ***.json
```

### Supported attack

| Score-Based Black-box attack|Function name| Paper| 
| :------------- |:-------------|:-----|
| NES Attack   | nes_attack.py NESAttack |Black-box Adversarial Attacks with Limited Queries and Information ICML 2018|
| ZO-signSGD  | zo_sign_agd_attack.py ZOSignSGDAttack |signSGD via Zeroth-Order Oracle ICLR 2019|
| Bandit Attack   | bandit_attack.py BanditAttack |Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors ICML 2019|
| SimBA   | simple_attack.py SimpleAttack |Simple Black-box Adversarial Attacks ICML 2019|
| Parsimonious Attack  | parsimonious_attack.py ParsimoniousAttack |Parsimonious Black-Box Adversarial Attacks via Efficient Combinatorial Optimization ICML 2019|
| Sign Hunter   | sign_attack.py SignAttack |Sign Bits Are All You Need for Black-Box Attacks ICLR 2020|
| Square Attack   | square_attack.py SquareAttack |Square Attack: a query-efficient black-box adversarial attack via random search ECCV 2020|
| Meta Square Attack| meta_square_attack.py |Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks NeurIPS 2021|



| Decision-Based Black-box attack|Function name| Paper| 
| :------------- |:-------------|:-----|
| Boundary Attack | boundary_attack.py BoundaryAttack |Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models ICLR 2017|
| OPT   | opt_attack.py OptAttack |Query-Efficient Hard-label Black-box Attack: An Optimization-based Approach ICLR 2019|
| Sign-OPT   | sign_opt_attack.py SignOPTAttack | Sign OPT: A Query Efficient Hard label Adversarial Attack ICLR 2020|
| Evolutionary Attack  | evo_attack.py EvolutionaryAttack |Efficient Decision based Blackbox Adversarial Attacks on Face Recognition CVPR 2019|
| GeoDA   | geoda_attack.py GeoDAttack |GeoDA: a geometric framework for blackbox adversarial attacks CVPR 2020|
| HSJA   | hsja_attack.py HSJAttack | HopSkipJumpAttack: A Query Efficient Decision Based Attack SP 2020|
| Sign Flip Attack   | sign_flip_attack.py SignFlipAttack |Boosting Decision based Blackbox Adversarial Attacks with Random Sign Flip ECCV 2020|
| RayS  | rays_attack.py RaySAttack | RayS: A Ray Searching Method for Hard-label Adversarial Attack KDD 2020|
| PSJA  | psja_attack.py PSJAAttack | PopSkipJump: Decision-Based Attack for Probabilistic Classifiers ICML 2021|   



### Supported dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [ImageNet](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack/overview/dataset). Here, we test the contained attack methods on the whole CIFAR-10 testing set and ImageNet competition dataset comprised of 1000 samples.

### Supported testing models

You can test all models trained on CIFAR-10 and ImageNet by adding loading code of your testing model in utils/model_loader.py.


### Attack result of query-based attacks (attack success rate, average number and median number of success attacks)

#### Score-based Attacks

##### ImageNet

###### linf = 0.05
|   Model  |    Metrics   | Bandit | NES    | Parsimonious | SignHunter | Square | ZOSignSGD |
|:--------:|:------------:|--------|--------|--------------|------------|--------|-----------|
|          |      ASR     | 0.98   | 1      | 1            | 0.97       | 1      | 0.97      |
| Resnet50 |  Mean Query  | 329.2  | 1031.9 | 347.7        | 261.4      | 76.5   | 2013.0    |
|          | Median Query | 58.0   | 720.0  | 241.0        | 85.0       | 13.0   | 1220.0    |

![avatar](/pics/score-based1.jpg)

###### l2 = 5
|   Model  |    Metrics   | Bandit | NES    | SimBA  | Square | ZOSignSGD |
|:--------:|:------------:|--------|--------|--------|--------|-----------|
|          |      ASR     | 1      | 0.96   | 0.69   | 0.99   | 0.46      |
| Resnet50 |  Mean Query  | 856.5  | 1335.2 | 1234.5 | 612.1  | 843.2     |
|          | Median Query | 512.0  | 1020.0 | 1120.0 | 174.0  | 549.0     |

![avatar](/pics/score-based2.jpg)

##### Cifar10

###### linf = 0.05
| Model |    Metrics   | Bandit | NES   | Parsimonious |  Sign | Square | ZOSignSGD |
|:-----:|:------------:|--------|-------|--------------|-----|--------|-----------|
|       |      ASR     | 1      | 1     | 1            |   1   | 1      | 1         |
|  VGG  |  Mean Query  | 111.3  | 267.3 | 207.5        | 106.8 | 73.3   | 257.4     |
|       | Median Query | 54.0   | 180.0 | 146.0        |  57.0 | 35.0   | 155.0     |
|       |      ASR     | 1      | 1     | 1            |   1   | 1      | 1         |
| wrn28 |  Mean Query  | 210.2  | 465.5 | 457.8        | 167.6 | 90.3   | 581.8     |
|       | Median Query | 72.0   | 210.0 | 201.0        |  74.0 | 28.0   | 217.0     |

![avatar](/pics/score-based3.jpg)

###### l2 = 1
| Model |    Metrics   | Bandit | NES   | SimBA | Square | ZOSignSGD |
|:-----:|:------------:|--------|-------|-------|--------|-----------|
|       |      ASR     | 1      | 1     | 1     | 1      | 1         |
|  VGG  |  Mean Query  | 393.0  | 374.8 | 308.0 | 435.0  | 340.1     |
|       | Median Query | 182.0  | 270.0 | 136.0 | 178.0  | 275.0     |
|       |      ASR     | 1      | 0.97  | 0.96  | 0.97   | 0.77      |
| wrn28 |  Mean Query  | 619.0  | 729.1 | 457.2 | 639.9  | 967.4     |
|       | Median Query | 260.0. | 360.0 | 190.0 | 200.0  | 372.0     |

![avatar](/pics/score-based4.jpg)


#### Decision-based Attacks

##### ImageNet

###### linf = 0.05
|   Model  |    Metrics   | GeoDA | HSJA | SignOPT | RayS | SignFlip |
|:--------:|:------------:|-------|------|---------|----|----------|
|          |      ASR     | 0.80  | 0.21 | 0.36    | 0.98 | 0.97     |
| Resnet50 |  Mean Query  | 1009  | 846  | 2399    | 1144 | 1548     |
|          | Median Query | 240   | 182  | 1980    |  595 | 732      |

![avatar](/pics/decision-based1.jpg)

###### l2 = 5
|   Model  |    Metrics   | GeoDA | HSJA | SignOPT | Evolutionary | Boundary |
|:--------:|:------------:|-------|------|---------|--------------|----------|
|          |      ASR     | 0.60  | 0.79 | 0.37    | 0.21           | 0.04        |
| Resnet50 |  Mean Query  | 1473  | 3339 | 2019    | 2156            | 2267        |
|          | Median Query | 612   | 2550 | 1598    | 598            | 339        |

![avatar](/pics/decision-based2.jpg)

##### Cifar10

###### linf = 0.05
| Model |    Metrics   | GeoDA | HSJA | SignOPT | RayS | SignFlip |
|:-----:|:------------:|-------|------|---------|----|----------|
|       |      ASR     | 0.96  | 1    | 0.96    |   1  | 1        |
|  VGG  |  Mean Query  | 760   | 780  | 1803    |  510 | 200      |
|       | Median Query | 347   | 557  | 1561    |  338 | 120      |
|       |      ASR     | 0.95  | 0.99 | 0.97    |   1  | 1        |
| wrn28 |  Mean Query  | 832   | 878  | 1955    |  683 | 216      |
|       | Median Query | 376   | 557  | 1656    |  453 | 136      |

![avatar](/pics/decision-based3.jpg)

###### l2 = 1
| Model |    Metrics   | GeoDA | HSJA | SignOPT | Evolutionary | Boundary |
|:-----:|:------------:|-------|------|---------|--------------|----------|
|       |      ASR     | 0.66  | 1    | 0.98    | 0.24            | 0.06        |
|  VGG  |  Mean Query  | 1355  | 1220 | 1317    | 1814            | 2147        |
|       | Median Query | 575   | 974  | 1061    | 1033            | 1766        |
|       |      ASR     | 0.69  | 1    | 1       | 0.26            | 0.05        |
| wrn28 |  Mean Query  | 1444  | 1208 | 1426    | 1933            | 2038        |
|       | Median Query | 580   | 961  | 1282    | 1211            | 1496        |

![avatar](/pics/decision-based4.jpg)
