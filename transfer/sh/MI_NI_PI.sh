#!/bin/bash -l

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "------------------------------------------"
echo "Craft adversarial examples by all method"
echo "Experiment setting"
echo "Dataset: NIPS2017"
echo "Attack: untargeted inf"
echo "Surrogate(single): resnet50 vgg19_bn densenet121 inc-v3 vit"
echo "------------------------------------------"

ATTACK="python -u main_attack.py"

SOURCE_PATH="NIPS2017/pretrained/resnet50"
TARGET_PATH="NIPS2017/pretrained/resnet50 NIPS2017/pretrained/vgg19_bn NIPS2017/pretrained/resnet152
        NIPS2017/pretrained/inception_v3 NIPS2017/pretrained/densenet121 NIPS2017/pretrained/mobilenet_v2
        NIPS2017/pretrained/senet154 NIPS2017/pretrained/resnext101 NIPS2017/pretrained/wrn101
        NIPS2017/pretrained/pnasnet NIPS2017/pretrained/mnasnet NIPS2017/pretrained/vit_b_16
        NIPS2017/pretrained/swin_b NIPS2017/pretrained/convnext_b NIPS2017/pretrained/adv_resnet50
        NIPS2017/pretrained/adv_swin_b NIPS2017/pretrained/adv_convnext_b"


PATH_CSV="./csv_files/MI_NI_PI.csv"
PATH_JSON_BASE="./json_files/NIPS2017/untargeted/l_inf"
NORM_STEPS=(0.0001 0.00005 0.00001)
BATCH_SIZE=500

for NORM_STEP in "${NORM_STEPS[@]}" ; do

  PATH_ADV_BASE="./adv_imgs/analysis/MI_NI_PI/$NORM_STEP"
  ARGS_COMMON="--epsilon 0.03 --norm-step $NORM_STEP --seed 0"

#  echo "------ MI-FGSM ------"
#  METHOD="MI-FGSM"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 400 --save-dir "${PATH_ADV_BASE}/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#  --decay-factor 1

  echo "------ NI ------"
  METHOD="NI"
  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 400 --save-dir "${PATH_ADV_BASE}/${METHOD}" \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

  echo "------ PI ------"
  METHOD="PI"
  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 400 --save-dir "${PATH_ADV_BASE}/${METHOD}" \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

done

#PATH_CSV="./csv_files/MI_NI_PI.csv"
#PATH_JSON_BASE="./json_files/NIPS2017/untargeted/l_inf"
#DECAY_FACTORS=(1 0.9 0.8 0.7 0.6 0.5 0.4)
#BATCH_SIZE=500
#
#for DECAY_FACTOR in "${DECAY_FACTORS[@]}" ; do
#
#  PATH_ADV_BASE="./adv_imgs/analysis/MI_NI_PI/$DECAY_FACTOR"
#  ARGS_COMMON="--epsilon 0.03 --norm-step 0.00392157 --seed 0"
#
#  echo "------ MI-FGSM ------"
#  METHOD="MI-FGSM"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 50 --save-dir "${PATH_ADV_BASE}/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#  --decay-factor $DECAY_FACTOR
#
#done
