#!/bin/bash -l

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=0,1.2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "------------------------------------------"
echo "Craft adversarial examples by all method on regular(adversarial training) models"
echo "Attack adversarial training(regular) models"
echo "Experiment setting"
echo "Dataset: NIPS2017"
echo "Attack: untargeted inf"
echo "Regular & adversarial training model architecture: (adv)resnet50 (adv)convnext_b (adv)swin_b"
echo "记得去掉eval的transform！！！！！！！！！！"
echo "------------------------------------------"

ATTACK="python -u main_attack.py"
ARGS_COMMON="--epsilon 0.03 --norm-step 0.00392157 --seed 0"
PATH_CSV="./csv_files/regular_at_NIPS2017_UT_INF.csv"
PATH_JSON_BASE="./json_files/NIPS2017/untargeted/l_inf"
PATH_ADV_BASE="./adv_imgs/analysis/regular_at_NIPS2017_UT_INF"


echo "-------------architecture: resnet50---------------"
SOURCES=(adv_swin_b)  # adv_resnet50 resnet50 convnext_b swin_b adv_resnet50 adv_convnext_b adv_swin_b
TARGET_PATH="NIPS2017/pretrained/resnet50 NIPS2017/pretrained/convnext_b NIPS2017/pretrained/swin_b
             NIPS2017/pretrained/adv_resnet50 NIPS2017/pretrained/adv_convnext_b NIPS2017/pretrained/adv_swin_b"

for SOURCE in "${SOURCES[@]}" ; do
  SOURCE_PATH="NIPS2017/pretrained/$SOURCE"

#  echo "------ I-FGSM ------"
#  METHOD="I-FGSM"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#  echo "------ MI-FGSM ------"
#  METHOD="MI-FGSM"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#  echo "------ NI ------"
#  METHOD="NI"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 10 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#  echo "------ PI ------"
#  METHOD="PI"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 10 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#  echo "------ VT ------"
#  METHOD="VT"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV

  echo "------ RAP ------"
  METHOD="RAP"
  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 400 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
  --input-transformation "add_reverse_perturbation(late_start=100)"
#
#  echo "------ DI2-FGSM ------"
#  METHOD="DI2-FGSM"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#  --input-transformation "DI(in_size=224, out_size=256)"
#
#  echo "------ SI ------"
#  METHOD="SI"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#  echo "------ admix ------"
#  METHOD="admix"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#  echo "------ TI ------"
#  METHOD="TI"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV

done
