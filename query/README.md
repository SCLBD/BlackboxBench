# Query-based attacks

### 💡Requirements

All Python libraries that query-based attacks in BlackboxBench depend on are listed in [`requirements.txt`](requirements.txt). You can run the following script to configurate necessary environment:

```
pip install -r requirements.txt
```

------

### 🤩 Quick start❕

#### 1️⃣ Load pretrained models

Before users run the main file [attack_cifar10.py](attack_cifar10.py) & [attack_imagenet.py](attack_imagenet.py), they need to load pretrained model with `.pth` file. The following part is an example of how to load `Wide-Resnet-28` pretrained on `CIFAR10`. Users need to put pretrained model file '`cifar_wrn_28.pth`' into '`pretrained_models/`' and change the file path accordingly in [utils/model_loader.py](utils/model_loader.py).

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

🔗 Download links of pretrained weights can be found in Supplementary Sec.II of [our paper](https://arxiv.org/abs/2312.16979). 

#### 2️⃣ Configurate the hyperparameters of attacks

Users can modify the configuration file (***.json) to run different attack methods on different models with $l_\infty$ norm or $l_2$ norm. The following part is about how to modify a config-json file as desired. Here is an example config-json file for `SignOpt Attack` on `Wide-Resnet-28` (`CIFAR10`dataset).

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

where `l_inf` represents $l_\infty$ norm perturbation and `l_2` represents $l_2$ norm perturbation.



#### 3️⃣ Run attacks

After modifying the attacks config files in [config-jsons](config-jsons) as desired, include config files of the considered attacks in [attack_cifar10.py](attack_cifar10.py) as follows (running attack on cifar-10 as an example):

```
python attack_cifar10.py ***.json
```

