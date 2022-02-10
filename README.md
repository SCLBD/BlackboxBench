
### Setup

### Running Query-based Attacks

After modifying the attacks config files in `config-jsons` as desired, include config files of the considered attacks in `attack_cifar10.py` as follows:

```
python attack_cifar10.py ***.json
```

### Supported attack

| Score-Based Black-box attack|Function name| Paper| 
| :------------- |:-------------|:-----|
| NES Attack   | nes_attack.py NESAttack |Black-box Adversarial Attacks with Limited Queries and Information| ICML 2018|
| ZO-signSGD  | zo_sign_agd_attack.py ZOSignSGDAttack |signSGD via Zeroth-Order Oracle| ICLR 2019|
| Bandit Attack   | bandit_attack.py BanditAttack |Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors| ICML 2019|
| SimBA   | simple_attack.py SimpleAttack |Simple Black-box Adversarial Attacks| ICML 2019|
| Parsimonious Attack  | parsimonious_attack.py ParsimoniousAttack |Parsimonious Black-Box Adversarial Attacks via Efficient Combinatorial Optimization| ICML 2019|
| Sign Hunter   | sign_attack.py SignAttack |Sign Bits Are All You Need for Black-Box Attacks| ICLR 2020|
| Square Attack   | square_attack.py SquareAttack |Square Attack: a query-efficient black-box adversarial attack via random search| ECCV 2020|
| Meta Square Attack| meta_square_attack.py |Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks| NeurIPS 2021|



| Decision-Based Black-box attack|Function name| Paper| 
| :------------- |:-------------|:-----|
| Boundary Attack | boundary_attack.py BoundaryAttack |Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models| ICLR 2017|
| OPT   | opt_attack.py OptAttack |Query-Efficient Hard-label Black-box Attack: An Optimization-based Approach| ICLR 2019|
| Sign-OPT   | sign_opt_attack.py SignOPTAttack | Sign OPT: A Query Efficient Hard label Adversarial Attack| ICLR 2020|
| Evolutionary Attack  | evo_attack.py EvolutionaryAttack |Efficient Decision based Blackbox Adversarial Attacks on Face Recognition|CVPR 2019|
| GeoDA   | geoda_attack.py GeoDAttack |GeoDA: a geometric framework for blackbox adversarial attacks| CVPR 2020|
| HSJA   | hsja_attack.py HSJAttack | HopSkipJumpAttack: A Query Efficient Decision Based Attack| SP 2020|
| Sign Flip Attack   | sign_flip_attack.py SignFlipAttack |Boosting Decision based Blackbox Adversarial Attacks with Random Sign Flip| ECCV 2020|
| RayS   | rays_attack.py RaySAttack | RayS: A Ray Searching Method for Hard-label Adversarial Attack| KDD 2020|



### Supported dataset
CIFAR-10, ImageNet

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
