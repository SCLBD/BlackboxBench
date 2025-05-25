## Transfer-based attacks

### 💡 Requirements

All Python libraries that tranfer-based attacks in BlackboxBench depend on are listed in [`requirements.txt`](requirements.txt). You can run the following script to configurate necessary environment:

```
pip install -r requirements.txt
```

------

### 🤩 Quick start❕

#### 1️⃣ Load pretrained models

Before user run the main file [main_attack.py](main_attack.py), they need to load model with `.pth` file. 

📍 If a <u>standard pretrained</u> model is desired 

Here is an example of how to load `ResNet-50` pretrained on `ImageNet`. Users need to put pretrained model file `resnet50-19c8e357.pth` into '`surrogate_model/NIPS2017/pretrained/`' and change the file path in the according model framework file [surrogate_model/imagenet_models/resnet.py](surrogate_model/imagenet_models/resnet.py): 

```
def resnet50(pretrained=False, **kwargs):
    if pretrained:
        state_dict_dir = './surrogate_model/NIPS2017/pretrained/resnet50-19c8e357.pth'
    else:
        state_dict_dir = None

    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], state_dict_dir, progress=True,**kwargs)
```

(🔗 Download links of pretrained weights can be found in Supplementary Sec. IV of [our paper](https://arxiv.org/abs/2312.16979). )

📍 If an <u>user-customized</u> model is desired

Here is an example of how to load an user-customized model `<MODEL_ARCH>` pretrained on `<DATASET>`. Users need to put model file `***.pth` into '`surrogate_model/<DATASET>/<MODEL_ARCH>/`'

Valid `<DATASET>` includes: 

```
['CIFAR10', 'NIPS2017']
```

Valid `<MODEL_ARCH>` includes: 

```
IMAGENET_MODEL_NAMES = ['googlenet', 'alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet152', 'resnext101', 'wrn50', 'wrn101', 'inception_v3', 'densenet121', 'densenet201', 'vgg11_bn', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'shufflenet_v2_x1_0', 'mobilenet_v2', 'mobilenet_v3', 'mobilenet_v3_large', 'squeezenet1_1', 'senet154', 'mnasnet', 'efficientnet_b7', 'regnet_y_16gf', 'convnext_b', 'convnext_l', 'convnext_t', 'convnext_t_tv', 'convnext_l_tv', 'vit_b_16', 'vit_s_16', 'vit_b_16_google', 'swin_b', 'swin_t', 'swin_l', 'swin_s', 'xcit_s', 'pnasnet5_l', 'deit_s', 'poolformer_s', 'pvt_b', 'tnt_s', 'cait_s', 'adv_resnet50_gelu', 'adv_xcit_s', 'adv_rawrn_101_2_Peng2023Robust', 'adv_wrn_50_2_Salman2020Do_50_2', 'adv_resnet50_Salman2020Do_R50', 'adv_resnet50_Engstrom2019Robustness', 'adv_resnet50_Wong2020Fast', 'adv_resnet18_Salman2020Do_R18', 'adv_convnext_l_Liu2023Comprehensive_ConvNeXt_L', 'adv_convnext_b_Liu2023Comprehensive_ConvNeXt_B', 'adv_convnext_l_convstem_Singh2023Revisiting_ConvNeXt_L_ConvStem', 'adv_convnext_b_convstem_Singh2023Revisiting_ConvNeXt_B_ConvStem', 'adv_convnext_s_convstem_Singh2023Revisiting_ConvNeXt_S_ConvStem', 'adv_convnext_t_convstem_Singh2023Revisiting_ConvNeXt_T_ConvStem', 'adv_swin_b_Liu2023Comprehensive_Swin_B', 'adv_swin_l_Liu2023Comprehensive_Swin_L', 'adv_xcit_m_Debenedetti2022Light_XCiT_M12', 'adv_xcit_l_Debenedetti2022Light_XCiT_L12', 'adv_vit_b_convstem_Singh2023Revisiting_ViT_B_ConvStem', 'adv_vit_s_convstem_Singh2023Revisiting_ViT_S_ConvStem', 'tf2torch_adv_inception_v3', 'tf2torch_ens3_adv_inc_v3', 'tf2torch_ens4_adv_inc_v3', 'tf2torch_ens_adv_inc_res_v2',]
CIFAR10_MODEL_NAMES = ['densenet', 'pyramidnet272', 'resnext', 'vgg19_bn', 'wrn', 'gdas', 'adv_wrn_28_10', 'resnet50', 'inception_v3']
```

#### 2️⃣ Configurate the hyperparameters of attacks

Users can modify the configuration file (***.json) to run different attack methods with   $\{l_\infty, l_2\} \times \{\text{targeted}, \text{untargeted}\}$ setting. Here is an example json file for $l_\infty$, untargeted `I-FGSM` with `ResNet-50` as the surrogate model on`NIPS2017`dataset, evaluated on three target models `VGG19_bn`, `ResNet-152`, `Inception-V3`.

```
{
  "source_model_path": ["NIPS2017/pretrained/resnet50"], #Path to all the model files of the ensembled surrogate models. Support path to a single model file or path containing many models.
  "target_model_path": ["NIPS2017/pretrained/vgg19_bn",
                        "NIPS2017/pretrained/resnet152",
                        "NIPS2017/pretrained/inception_v3"], #Path to all the target models.Only support path to a single model file.
  "n_iter": 100, #Number of iterations.
  "shuffle": true, #Random order of models vs sequential order of (ensembled) surrogate models.
  "batch_size": 200, #Batch size. Try a lower value if out of memory.
  "norm_type": "inf", #Type of L-norm.
  "epsilon": 0.03, #Max L-norm of the perturbation.
  "norm_step": 0.00392157, #Max norm at each step.
  "seed": 0, #Set random seed.
  "n_ensemble": 1, #Number of samples to ensemble for each iteration(Default: 1).
  "targeted": false, #Achieve targeted attack or not.
  "save_dir": "./save", #Path to save adversarial images.

  "input_transformation": "", #Input transformation compatible with each attack.
  "loss_function": "cross_entropy", #Loss function compatible with each attack.
  "grad_calculation": "general", #Define a gradient calculator compatible with each attack.
  "backpropagation": "nonlinear", #Linear backpropagation vs noninear backpropagation
  "update_dir_calculation": "sgd" #Update direction calculator compatible with each attack.
}
```

📍 If ensemble attacks is desired, list all ensembles models in `source_model_path` like this

```
"source_model_path": ["NIPS2017/pretrained/resnet50",
                      "NIPS2017/pretrained/wrn101",
                      "NIPS2017/pretrained/pnasnet",
                      "NIPS2017/pretrained/mnasnet",]
```

#### 3️⃣ Run attacks

After modifying the attacks config files as desired, include config files of the considered attacks in [main_attack.py](main_attack.py) as follows (running [config/NIPS2017/untargeted/l_inf/I-FGSM.json](config/NIPS2017/untargeted/l_inf/I-FGSM.json) as an example):

```
python -u main_attack.py --json-path ./config/NIPS2017/untargeted/l_inf/I-FGSM.json
```

To fully reproduce the evalutions in [BlackboxBench](https://arxiv.org/abs/2312.16979), please run the following `.sh` files

| NIPS2017   | Untargetd                                             | Targeted                                            |
| ---------- | ----------------------------------------------------- | --------------------------------------------------- |
| $l_\infty$ | [main_NIPS2017_UT_INF.sh](sh/main_NIPS2017_UT_INF.sh) | [main_NIPS2017_T_INF.sh](sh/main_NIPS2017_T_INF.sh) |
| $l_2$      | [main_NIPS2017_UT_2.sh](sh/main_NIPS2017_UT_2.sh)     | [main_NIPS2017_T_2.sh](sh/main_NIPS2017_T_2.sh)     |

------



### 💡 Refined models

Transfer-based black-box attacks from **Model Perspective** refine the basic surrogate model to improve the transferability. If users wish to avoid refineing models on their own, we provide our pretrained checkpoints for LGV, SWA, Bayesian attack on CIFAR10 and NIPS2017:

| CIFAR10         | ResNet-50                                                    | VGG19-bn                                                     | Inception-V3                                                 | DenseNet-BC                                                  |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LGV             | [resnet50](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/El9GtTMpllZAhTVt02F8hikBXQGuP5vwVDtk8P338rYMBg?e=TAqef1) | [vgg19_bn](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/Em_NJJs-2qNPjAsm6CdzOFUBCZPII_RSExgzsz4WBUEO7A?e=eQHoqE) | [inception_v3](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EqvpMGUOod5GncDS96V9PSIBXK8_xSJ-YIR3f4uaaKj99g?e=WTbDjI) | [densenet](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EiJ-Il44_g9PtRxeygXhq6MBwboM5HXALuLbsP5Ya-3BlA?e=iledvh) |
| SWA             | [resnet50](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/Etoy6o_-GHpCmUDWqRuETHsBZZGVOwHWOwVRhimxTI5RyQ?e=YX9Qkk) | [vgg19_bn](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EmkjZ4Sdc-hIgtUErL7KJEgBuagA2XOskLoufUCme4AqNA?e=toxf8g) | [inception_v3](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EgsVSBX-GypMr7GQZEQ48j8BEGuTkWJalcE7WHd5I_NHWQ?e=po97AM) | [densenet](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EoMlIGGLlsVAlwvYpmbJ7vwBKomywucWbv--ZKMLfuw9ag?e=600rry) |
| Bayesian attack | [resnet50](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EpytPgSZUINHp6Pwd66yj4gBdLsQcaw8YyZmjrHdZaanEg?e=bZ3sH3) | [vgg19_bn](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EvTG68KdyxJOoHtvgnSEupsBiTC40FrIaY-U9eT02arNCQ?e=jh6qAa) | [inception_v3](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EsSoe_-H-R1NvcyErxGEa0oBFM__Rw12zO6Ean2PDbtvNg?e=NlPK7t) | [densenet](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/El306wGwBWlLkrGrGo2FtP0BFzDfbqOkIbsXHM8B0pQMzA?e=kYtXei) |

| NIPS2017        | ResNet-50                                                    | VGG19-bn                                                     | Inception-V3                                                 | DenseNet-121                                                 | ViT-B/16                                                     |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LGV             | [resnet50](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EjNj6HsDfzpPtTSoRF_Id-IByOTeg6PySiG7XanykNUg3Q?e=U1KUxC) | [vgg19_bn](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/ErM0lA5WGSRIgv64X9DNiK4BGHKIDjwMwg8H4E0HlEgpjA?e=9JYzB9) | [inception_v3](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EsJWohHeku9NnfCuJkOWMKwBu2RQXJ-xLhRpdsSO-_5TcA?e=1TLPVu) | [densenet121](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EoP5uQKPJh5HhSHMQdo99G4BZvMjJhn7KI2aFzdE52EUfg?e=2OQeEA) | [vit_b_16](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EkfhpJgXCTFGhncJFJzrysABeUEuta5he5NKZT35CZpKKA?e=k8sEL1) |
| SWA             | [resnet50](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EmG7e2sDKoZKmDtR3T3aBecBE6yOt8ImXhXCVu7X0ETIoQ?e=8eh2pj) | [vgg19_bn](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/Epgl95VHDNhMtrP19mZU5YoBhawHn3AJ1Q4755femtGplw?e=xnuKbL) | [inception_v3](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/Eqje5nbSr3FBhiLDaneELZIBD1ZXl__n3OJnZ9yi1CdLtA?e=gHa7cp) | [densenet121](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EkuStCpJdINNg-v-Ur4iR_sBKfKGsXgbja2o0DS1gOUelQ?e=h8Y0XC) | [vit_b_16](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EneXuvPYdfdMusAPwn3sfeYBr0znmzJELSslfRT0pl7hMw?e=HHkC9J) |
| Bayesian attack | [resnet50](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/ElzIHJ24hxFLhE2sJwm1OokB-dYDn_aVNYlW4NA4bcCJVQ?e=ixXx7I) | [vgg19_bn](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EgawsIhLoW9Goe71s5aUF40B24L8A5WsNRlIWRL-p8UXzQ?e=dutLl5) | [inception_v3](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/Eo6BaOHTzndNtNJMhnUYW9cB1h38aui8WaUZ4Qu4novGDQ?e=rn3DyF) | [densenet121](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/Eq3dtBJ61atBm9cwt_DHhdwBgzkUTCr746996-s0F958cw?e=JCbdtg) | [vit_b_16](https://cuhko365-my.sharepoint.com/:f:/g/personal/223040254_link_cuhk_edu_cn/EncCiO8A46RMh7SCJ3wpgToB0TY6C-ida33xCe47rRUKtg?e=kXD1AR) |

DRA models can be downloaded from [DRA repository](https://github.com/alibaba/easyrobust/tree/main/examples/attacks/dra).

------

### 💡 Acknowledgements

The following excellent resources are very helpful for our work. Please consider leaving a ⭐ on their repositories.

Codes:

https://github.com/Framartin/lgv-geometric-transferability/tree/main?tab=readme-ov-file

https://github.com/qizhangli/linbp-attack

https://github.com/SCLBD/Transfer_attack_RAP/tree/main

https://github.com/ZhengyuZhao/TransferAttackEval/tree/main

Pretrained weights:

https://pytorch.org/vision/stable/models.html

https://www.kaggle.com/datasets/firuzjuraev/trained-models-for-cifar10-dataset?resource=download

https://github.com/bearpaw/pytorch-classification

https://github.com/Cadene/pretrained-models.pytorch

https://github.com/D-X-Y/AutoDL-Projects

https://github.com/ZiangYan/subspace-attack.pytorch
