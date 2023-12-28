# BlackboxBench: a comprehensive benchmark of adversarial black-box attack methods

![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic)
![Pytorch 2.0.1](https://img.shields.io/badge/pytorch-2.0.1-DodgerBlue.svg?style=plastic)
![CUDA 11.8](https://img.shields.io/badge/cuda-11.8-DodgerBlue.svg?style=plastic)
<!--- ![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic) -->

BlackboxBench is a comprehensive benchmark containing mainstream adversarial black-box attack methods implemented based on [PyTorch](https://pytorch.org). 
It can be used to evaluate the adversarial robustness of any ML models, or as the baseline to develop more advanced attack and defense methods. 
Currently, we support:

- Attack methods: 
	- Query-based attack methods: 
		- `7 score-based attacks`: [NES](https://arxiv.org/abs/1804.08598), [ZOSignSGD](https://openreview.net/forum?id=BJe-DsC5Fm), [Bandit-prior](https://arxiv.org/abs/1807.07978), [ECO attack](https://arxiv.org/abs/1905.06635), [SimBA](https://arxiv.org/abs/1905.07121), [SignHunter](https://openreview.net/forum?id=SygW0TEFwH), [Sqaure attack](https://arxiv.org/abs/1912.00049).
		- `8 decision-based attacks`: [Boundary attack](https://arxiv.org/abs/1712.04248), [OPT attack](https://arxiv.org/abs/1807.04457), [Sign-OPT](https://arxiv.org/abs/1909.10773), [Evoluationary attack](https://arxiv.org/abs/1904.04433), [GeoDA](https://arxiv.org/abs/2003.06468), [HSJA](https://arxiv.org/abs/1904.02144), [Sign Flip](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2336_ECCV_2020_paper.php), [RayS](https://arxiv.org/abs/2006.12792).
	- Transfer attack methods: `Coming soon!`
- Datasets: CIFAR-10, ImageNet.
- Models: Several CNN models pretrained on above two datasets.

We also provide a [**public leaderboard**](https://blackboxbench.github.io/index.html) of evaluating above black-box attack performance against several undefended and defended deep models, on above two datasets.

BlackBoxBench will be continously updated by adding implementations of more attack and defense methods, as well as evaluations on more datasets and models. **You are welcome to contribute your black-box attack methods to BlackBoxBench.**

---
<font size=5><center><b> Table of Contents </b> </center></font>

- [Requirements](#requirement)
- [Usage](#usage)
  - [Load pretrained models](#load-pretrained-models) 
  - [Configurate the hyperparameters of attacks](#set-the-hyperparameters-of-attacks)
  - [Run Attacks](#run-attacks)
- [Supported attacks](#supported-attacks)
- [Supported datasets](#supported-datasets)
- [Supported testing models](#supported-testing-models)
- [Supported defense methods](#supported-defense-methods)
- [Citation](#citation)


---

### Requirements:

You can run the following script to configurate necessary environment


```
pip install -r requirement.txt
```
---
### Usage:

#### 1. Load pretrained models
Before users run the main file [attack_cifar10.py](./attack_cifar10.py) & [attack_imagenet.py](./attack_imagenet.py), they need to load pretrained model with `.pth` file. The following part is an example of how to load `Wide-Resnet-28` pretrained on `CIFAR10`. Users need to put pretrained model file '`cifar_wrn_28.pth`' into '`pretrained_models/`' and change the file path accordingly in [utils/model_loader.py](./utils/model_loader.py).

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


#### 2. Configurate the hyperparameters of attacks.

Users can modify the configuration file (***.json) to run different attack methods on different models with l_inf norm or l_2 norm. The following part is about how to modify a config-json file as desired. Here is an example config-json file for `Signopt Attack` on `Wide-Resnet-28` (`CIFAR10`dataset).
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
ImageNet: l_inf: 0.05 =  12.75/255, l_2: 5 = 1275/255	
```

where `l_inf` represents l_infty norm perturbation and `l_2` represents l_2 norm perturbation.


#### 3. Run Attacks

After modifying the attacks config files in [config-jsons](./config-jsons) as desired, include config files of the considered attacks in [attack_cifar10.py](./attack_cifar10.py) as follows (running attack on cifar-10 as an example):

```
python attack_cifar10.py ***.json
```
---
### Supported attacks

| Score-Based Black-box attack|File name| Paper| 
| :------------- |:-------------|:-----|
| NES Attack   | [nes_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/score-based%20attacks/nes_attack.py) |[Black-box Adversarial Attacks with Limited Queries and Information](https://arxiv.org/abs/1804.08598) ICML 2018|
| ZO-signSGD  | [zo_sign_agd_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/score-based%20attacks/zo_sign_sgd_attack.py)  |[signSGD via Zeroth-Order Oracle](https://openreview.net/forum?id=BJe-DsC5Fm) ICLR 2019|
| Bandit Attack   | [bandit_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/score-based%20attacks/bandit_attack.py) |[Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors](https://arxiv.org/abs/1807.07978) ICML 2019|
| SimBA   | [simple_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/score-based%20attacks/simple_attack.py) |[Simple Black-box Adversarial Attacks](https://arxiv.org/abs/1905.07121) ICML 2019|
| ECO Attack  | [parsimonious_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/score-based%20attacks/parsimonious_attack.py) |[Parsimonious Black-Box Adversarial Attacks via Efficient Combinatorial Optimization](https://arxiv.org/abs/1905.06635) ICML 2019|
| Sign Hunter   | [sign_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/score-based%20attacks/sign_attack.py) |[Sign Bits Are All You Need for Black-Box Attacks](https://openreview.net/forum?id=SygW0TEFwH) ICLR 2020|
| Square Attack   | [square_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/score-based%20attacks/square_attack.py) |[Square Attack: a query-efficient black-box adversarial attack via random search](https://arxiv.org/abs/1912.00049) ECCV 2020|



| Decision-Based Black-box attack|File name| Paper| 
| :------------- |:-------------|:-----|
| Boundary Attack | [boundary_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/decision-based%20attacks/boundary_attack.py) |[Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models](https://arxiv.org/abs/1712.04248) ICLR 2017|
| OPT   | [opt_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/decision-based%20attacks/opt_attack.py) |[Query-Efficient Hard-label Black-box Attack: An Optimization-based Approach](https://arxiv.org/abs/1807.04457) ICLR 2019|
| Sign-OPT   | [sign_opt_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/decision-based%20attacks/sign_opt_attack.py) | [Sign OPT: A Query Efficient Hard label Adversarial Attack](https://arxiv.org/abs/1909.10773) ICLR 2020|
| Evolutionary Attack  | [evo_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/decision-based%20attacks/evo_attack.py) |[Efficient Decision based Blackbox Adversarial Attacks on Face Recognition](https://arxiv.org/abs/1904.04433) CVPR 2019|
| GeoDA   | [geoda_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/decision-based%20attacks/geoda_attack.py) |[GeoDA: a geometric framework for blackbox adversarial attacks](https://arxiv.org/abs/2003.06468) CVPR 2020|
| HSJA   | [hsja_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/decision-based%20attacks/hsja_attack.py) | [HopSkipJumpAttack: A Query Efficient Decision Based Attack](https://arxiv.org/abs/1904.02144) IEEE S&P 2020|
| Sign Flip Attack   | [sign_flip_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/decision-based%20attacks/sign_flip_attack.py) |[Boosting Decision based Blackbox Adversarial Attacks with Random Sign Flip](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2336_ECCV_2020_paper.php) ECCV 2020|
| RayS  | [rays_attack.py](https://github.com/SCLBD/B-box/blob/main/attacks/decision-based%20attacks/rays_attack.py) | [RayS: A Ray Searching Method for Hard-label Adversarial Attack](https://arxiv.org/abs/2006.12792) KDD 2020|



### Supported datasets
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [ImageNet](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack/overview/dataset). Please first download these two datasets into `data/`. Here, we test the contained attack methods on the whole CIFAR-10 testing set and ImageNet competition dataset comprised of [1000 samples](https://drive.google.com/file/d/1QGldLJtVXCU_hG6XYY5ju48W4X1ioA7l/view?usp=sharing).

### Supported testing models

You can test all models trained on CIFAR-10 and ImageNet by adding loading code of your testing model in [utils/model_loader.py](./utils/model_loader.py).
Here, we test the contained attack methods on the below models.
+ **CIFAR-10**: ResNet-50, WideResNet-28, AT-l_inf-WideResNet-28 [(with extra data (Gowal et al., 2020))](https://arxiv.org/abs/2010.03593), AT-l_inf-WideResNet-28 [(with data from DDPM (Rebuffi et al., 2021))](https://arxiv.org/abs/2103.01946).
For ResNet-50 and WideResNet-28, we train them by using the code from this [github repo](https://github.com/kuangliu/pytorch-cifar). 

+ **ImageNet**: ResNet-50, Inception-v3, AT-l_inf-ResNet-50 (4/255) [(Salman et al., 2020)](https://github.com/microsoft/robust-models-transfer), FastAT-l_inf-ResNet-50 (4/255) [(Wong et al., 2020)](https://github.com/locuslab/fast_adversarial).
For ResNet-50 and Inception-v3, we use the provided pretrained model from [torchvision](https://pytorch.org/vision/stable/index.html).

### Supported defense methods
Here, we also provide several defense methods against black-box attacks. 

+ **Random Noise Defense Against Query-Based Black-Box Attacks (RND) [(Qin et al., 2021)](https://arxiv.org/abs/2104.11470)**: RND is a lightweight and plug and play defense method against query-based attacks. It is realized by adding a random noise to each query at the inference time (one line code in Pytorch: x = x + noise_size * torch.randn like(x)). You can just tune the sigma (noise_size) to conduct RND in [attack_cifar10.py](./attack_cifar10.py) & [attack_imagenet.py](./attack_imagenet.py).


## [Citation](#citation)

<a href="#top">[Back to top]</a>

If you want to use this library in your research, cite it as
follows:

```
   @misc{blackboxbench,
      title={BlackboxBench (Python Library)},
      author={Zeyu Qin and Xuanchen Yan and Baoyuan Wu},
      year={2022},
      url={https://github.com/SCLBD/BlackboxBench}
   }

```


If interested, you can read our recent works about black-box attack and defense methods, and more works about trustworthy AI can be found [here](https://sites.google.com/site/baoyuanwu2015/home).

```
@inproceedings{cgattack-cvpr2022,
  title={Boosting Black-Box Attack with Partially Transferred Conditional Adversarial Distribution},
  author={Feng, Yan and Wu, Baoyuan and Fan, Yanbo and Liu, Li and Li, Zhifeng and Xia, Shutao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

@article{rnd-blackbox-defense-nips2021,
  title={Random Noise Defense Against Query-Based Black-Box Attacks},
  author={Qin, Zeyu and Fan, Yanbo and Zha, Hongyuan and Wu, Baoyuan},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@inproceedings{liang2021parallel,
  title={Parallel Rectangle Flip Attack: A Query-Based Black-Box Attack Against Object Detection},
  author={Liang, Siyuan and Wu, Baoyuan and Fan, Yanbo and Wei, Xingxing and Cao, Xiaochun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7697--7707},
  year={2021}
}

@inproceedings{chen2020boosting,
  title={Boosting decision-based black-box adversarial attacks with random sign flip},
  author={Chen, Weilun and Zhang, Zhaoxiang and Hu, Xiaolin and Wu, Baoyuan},
  booktitle={European Conference on Computer Vision},
  pages={276--293},
  year={2020},
  organization={Springer}
}

@inproceedings{evolutionary-blackbox-attack-cvpr2019,
  title={Efficient decision-based black-box adversarial attacks on face recognition},
  author={Dong, Yinpeng and Su, Hang and Wu, Baoyuan and Li, Zhifeng and Liu, Wei and Zhang, Tong and Zhu, Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7714--7722},
  year={2019}
}
```

## [Contributors](#contributors)

[Zeyu Qin](https://github.com/Alan-Qin), Run Liu, Xuanchen Yan

## [Copyright](#copyright)

<a href="#top">[Back to top]</a>

The source code of this repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).


This project is built by the Secure Computing Lab of Big Data ([SCLBD](http://scl.sribd.cn/index.html)) at The Chinese University of Hong Kong, Shenzhen and Shenzhen Research Institute of Big Data, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.



