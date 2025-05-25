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

SOURCES=(resnet50)
ENSEMBLE_PATH="NIPS2017/pretrained/resnet50 NIPS2017/pretrained/vgg19_bn NIPS2017/pretrained/densenet121
               NIPS2017/pretrained/inception_v3 NIPS2017/pretrained/vit_b_16"
ENSEMBLE_PATH_SGM="NIPS2017/pretrained/resnet50 NIPS2017/pretrained/densenet121 NIPS2017/pretrained/vit_b_16"
TARGET_PATH="NIPS2017/pretrained/resnet50 NIPS2017/pretrained/vgg19_bn NIPS2017/pretrained/resnet152
        NIPS2017/pretrained/inception_v3 NIPS2017/pretrained/densenet121 NIPS2017/pretrained/mobilenet_v2
        NIPS2017/pretrained/senet154 NIPS2017/pretrained/resnext101 NIPS2017/pretrained/wrn101
        NIPS2017/pretrained/pnasnet5_l NIPS2017/pretrained/mnasnet NIPS2017/pretrained/vit_b_16
        NIPS2017/pretrained/swin_b NIPS2017/pretrained/convnext_b NIPS2017/pretrained/adv_resnet50_Salman2020Do_R50
        NIPS2017/pretrained/adv_swin_b_Liu2023Comprehensive_Swin_B NIPS2017/pretrained/adv_convnext_b_Liu2023Comprehensive_ConvNeXt_B"

BATCH_SIZE=500
ARGS_COMMON="--epsilon 0.03 --norm-step 0.00392157 --seed 0 --norm-type inf " # untargeted linf
PATH_CSV="./csv_files/composite_NIPS2017_UT_INF_RESNET50_v1.csv"
PATH_ADV_BASE="./adv_imgs/analysis/composite_NIPS2017_UT_INF_RESNET50_v1"
PATH_JSON_BASE="./json_files/NIPS2017/untargeted/l_inf"


for SOURCE in "${SOURCES[@]}" ; do
  SOURCE_PATH="NIPS2017/pretrained/$SOURCE"

#  echo "------ TI-FGSM ------"
#  METHOD=TI-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "TI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ TI-DI-FGSM ------"
#  METHOD=TI-DI-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)|TI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ TI-SI-FGSM ------"
#  METHOD=TI-SI-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)|TI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ TI-Admix-FGSM ------"
#  METHOD=TI-Admix-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)|TI" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ TI-SIA-FGSM ------"
#  METHOD=TI-SIA-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)|TI" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DI-SIA-FGSM ------"
#  METHOD=DI-SIA-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)|DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DI-SIA-FGSM ------"
#  METHOD=DI-SIA-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)|siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SI-SIA-FGSM ------"
#  METHOD=SI-SIA-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)|SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SI-SIA-FGSM ------"
#  METHOD=SI-SIA-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)|siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ Admix-SIA-FGSM ------"
#  METHOD=Admix-SIA-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)|admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ Admix-SIA-FGSM ------"
#  METHOD=Admix-SIA-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)|siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#
#  echo "------ TI-MI-FGSM ------"
#  METHOD=TI-MI-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "TI" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ TI-MI-FGSM ------"
#  METHOD=TI-MI-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "TI" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ TI-NI-FGSM ------"
#  METHOD=TI-NI-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI|TI" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ TI-NI-FGSM ------"
#  METHOD=TI-NI-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI|TI" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ TI-VT-FGSM ------"
#  METHOD=TI-VT-FGSM
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "TI" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning"  --n-var-sample 5
#
#  echo "------ TI-RAP-FGSM ------"
#  METHOD="TI-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)|TI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ TI-LinBP-FGSM ------"
#  METHOD="TI-LinBP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "TI" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "sgd"
#
#  echo "------ TI-SGM-FGSM ------"
#  METHOD="TI-SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "TI" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ TI-FIA-FGSM ------"
#  METHOD="TI-FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "TI" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ TI-NAA-FGSM ------"
#  METHOD="TI-NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "TI" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ TI-DRA-FGSM ------"
#  METHOD="TI-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "TI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ TI-GhostNet-FGSM ------"
#  METHOD="TI-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "TI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ TI-LGV ------"
#  METHOD="TI-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "TI" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
  echo "------ Bayesian-FGSM ------"
  METHOD="Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"

  echo "------ TI-Bayesian-FGSM ------"
  METHOD="TI-Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
  --input-transformation "TI" --loss-function "cross_entropy" --grad-calculation "general"\
  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ DI-FGSM ------"
#  METHOD="DI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DI-SI-FGSM ------"
#  METHOD="DI-SI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)|DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DI-admix-FGSM ------"
#  METHOD="DI-admix-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)|DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DI-MI-FGSM ------"
#  METHOD="DI-MI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ DI-MI-FGSM ------"
#  METHOD="DI-MI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ DI-NI-FGSM ------"
#  METHOD="DI-NI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI|DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#    echo "------ DI-NI-FGSM ------"
#  METHOD="DI-NI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI|DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ DI-VT-FGSM ------"
#  METHOD="DI-VT-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning" --n-var-sample 5
#
#  echo "------ DI-RAP-FGSM ------"
#  METHOD="DI-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)|add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DI-RAP-FGSM ------"
#  METHOD="DI-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)|DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DI-PGN-FGSM ------"
#  METHOD="DI-PGN-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "PGN_grad(zeta=3, delta=0.5, N=20)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DI-LinBP-FGSM ------"
#  METHOD="DI-LinBP-FGSM"
#    $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 200 --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "sgd"
#
#  echo "------ DI-SGM-FGSM ------"
#  METHOD="DI-SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DI-FIA-FGSM ------"
#  METHOD="DI-FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DI-NAA-FGSM ------"
#  METHOD="DI-NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DI-DRA-FGSM ------"
#  METHOD="DI-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DI-GhostNet-EGSM ------"
#  METHOD="DI-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ DI-LGV-FGSM ------"
#  METHOD="DI-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
  echo "------ DI-Bayesian-FGSM ------"
  METHOD="DI-Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
  --backpropagation "nonlinear" --update-dir-calculation "sgd"\
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ DRA-FGSM ------"
#  METHOD="DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SI-FGSM ------"
#  METHOD="SI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SI-Admix-FGSM ------"
#  METHOD="SI-Admix-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)|SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SI-MI-FGSM ------"
#  METHOD="SI-MI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ SI-MI-FGSM ------"
#  METHOD="SI-MI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ SI-NI-FGSM ------"
#  METHOD="SI-NI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI|SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ SI-NI-FGSM ------"
#  METHOD="SI-NI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI|SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ SI-VT-FGSM ------"
#  METHOD="SI-VT-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning" --n-var-sample 5
#
#  echo "------ SI-RAP-FGSM ------"
#  METHOD="SI-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)|SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SI-RAP-FGSM ------"
#  METHOD="SI-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)|add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SI-PGN-FGSM ------"
#  METHOD="SI-PGN-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "PGN_grad(zeta=3, delta=0.5, N=20)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SI-LinBP-FGSM ------"
#  METHOD="SI-LinBP-FGSM"
#    $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "sgd"
#
#  echo "------ SI-SGM-FGSM ------"
#  METHOD="SI-SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2) "\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SI-FIA-FGSM ------"
#  METHOD="SI-FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SI-NAA-FGSM ------"
#  METHOD="SI-NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SI-DRA-FGSM ------"
#  METHOD="SI-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SI-GhostNet-FGSM ------"
#  METHOD="SI-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ SI-LGV-FGSM ------"
#  METHOD="SI-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
  echo "------ SI-Bayesian-FGSM ------"
  METHOD="SI-Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
  --input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general" \
  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ Admix-FGSM ------"
#  METHOD="Admix-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ Admix-MI-FGSM ------"
#  METHOD="Admix-MI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ Admix-MI-FGSM ------"
#  METHOD="Admix-MI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ Admix-NI-FGSM ------"
#  METHOD="Admix-NI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI|admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ Admix-NI-FGSM ------"
#  METHOD="Admix-NI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI|admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ Admix-VT-FGSM ------"
#  METHOD="Admix-VT-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning" --n-var-sample 5
#
#  echo "------ Admix-RAP-FGSM ------"
#  METHOD="Admix-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)|admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ Admix-RAP-FGSM ------"
#  METHOD="Admix-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)|add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ Admix-PGN-FGSM ------"
#  METHOD="Admix-PGN-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "PGN_grad(zeta=3, delta=0.5, N=20)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ Admix-LinBP-FGSM ------"
#  METHOD="Admix-LinBP-FGSM"
#    $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "sgd"
#
#  echo "------ Admix-SGM-FGSM ------"
#  METHOD="Admix-SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2) " \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ Admix-FIA-FGSM ------"
#  METHOD="Admix-FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ Admix-NAA-FGSM ------"
#  METHOD="Admix-NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ Admix-DRA-FGSM ------"
#  METHOD="Admix-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ Admix-GhostNet-FGSM ------"
#  METHOD="Admix-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ Admix-LGV-FGSM ------"
#  METHOD="Admix-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
  echo "------ Admix-Bayesian-FGSM ------"
  METHOD="Admix-Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general" \
  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ SIA-FGSM ------"
#  METHOD="SIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SIA-MI-FGSM ------"
#  METHOD="SIA-MI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ SIA-MI-FGSM ------"
#  METHOD="SIA-MI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ SIA-NI-FGSM ------"
#  METHOD="SIA-NI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI|siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ SIA-NI-FGSM ------"
#  METHOD="SIA-NI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI|siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ SIA-VT-FGSM ------"
#  METHOD="SIA-VT-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning" --n-var-sample 5
#
#  echo "------ SIA-RAP-FGSM ------"
#  METHOD="SIA-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)|siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SIA-RAP-FGSM ------"
#  METHOD="SIA-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)|add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SIA-PGN-FGSM ------"
#  METHOD="SIA-PGN-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 50 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "PGN_grad(zeta=3, delta=0.5, N=20)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SIA-LinBP-FGSM ------"
#  METHOD="SIA-LinBP-FGSM"
#    $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 10 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "sgd"
#
#  echo "------ SIA-SGM-FGSM ------"
#  METHOD="SIA-SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2) " \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SIA-FIA-FGSM ------"
#  METHOD="SIA-FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 10 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SIA-NAA-FGSM ------"
#  METHOD="SIA-NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 10 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SIA-DRA-FGSM ------"
#  METHOD="SIA-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SIA-GhostNet-FGSM ------"
#  METHOD="SIA-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ SIA-LGV-FGSM ------"
#  METHOD="SIA-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
  echo "------ SIA-Bayesian-FGSM ------"
  METHOD="SIA-Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general" \
  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ MI-FGSM ------"
#  METHOD="MI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ MI-VT-FGSM ------"
#  METHOD="MI-VT-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning|momentum" --n-var-sample 5 --decay-factor 1
#
#  echo "------ MI-RAP-FGSM ------"
#  METHOD="MI-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ MI-PGN-FGSM ------"
#  METHOD="MI-PGN-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "PGN_grad(zeta=3, delta=0.5, N=20)"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ MI-LinBP-FGSM ------"
#  METHOD="MI-LinBP-FGSM"
#    $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ MI-SGM-FGSM ------"
#  METHOD="MI-SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2) " \
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ MI-FIA-FGSM ------"
#  METHOD="MI-FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ MI-NAA-FGSM ------"
#  METHOD="MI-NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ MI-DRA-FGSM ------"
#  METHOD="MI-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ MI-GhostNet-FGSM ------"
#  METHOD="MI-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --ghost-attack --decay-factor 1
#
#  echo "------ MI-LGV-FGSM ------"
#  METHOD="MI-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1\
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
#  echo "------ MI-Bayesian-FGSM ------"
#  METHOD="MI-Bayesian-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1\
#  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
#  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ MI-FGSM ------"
#  METHOD="MI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ MI-VT-FGSM ------"
#  METHOD="MI-VT-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning|momentum" --n-var-sample 5 --decay-factor 1
#
#  echo "------ MI-LinBP-FGSM ------"
#  METHOD="MI-LinBP-FGSM"
#    $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ MI-SGM-FGSM ------"
#  METHOD="MI-SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2) " \
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ MI-FIA-FGSM ------"
#  METHOD="MI-FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ MI-NAA-FGSM ------"
#  METHOD="MI-NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ MI-DRA-FGSM ------"
#  METHOD="MI-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ MI-GhostNet-FGSM ------"
#  METHOD="MI-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --ghost-attack --decay-factor 1
#
#  echo "------ MI-LGV-FGSM ------"
#  METHOD="MI-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1\
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
  echo "------ MI-Bayesian-FGSM ------"
  METHOD="MI-Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general" \
  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1\
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ NI-FGSM ------"
#  METHOD="NI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ NI-VT-FGSM ------"
#  METHOD="NI-VT-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning|momentum" --n-var-sample 5 --decay-factor 1
#
#  echo "------ NI-RAP-FGSM ------"
#  METHOD="NI-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI|add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ NI-RAP-FGSM ------"
#  METHOD="NI-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)|look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ NI-PGN-FGSM ------"
#  METHOD="NI-PGN-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "PGN_grad(zeta=3, delta=0.5, N=20)"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ NI-LinBP-FGSM ------"
#  METHOD="NI-LinBP-FGSM"
#    $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ NI-SGM-FGSM ------"
#  METHOD="NI-SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2) "\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ NI-FIA-FGSM ------"
#  METHOD="NI-FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ NI-NAA-FGSM ------"
#  METHOD="NI-NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ NI-DRA-FGSM ------"
#  METHOD="NI-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ NI-GhostNet-FGSM ------"
#  METHOD="NI-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --ghost-attack --decay-factor 1
#
#  echo "------ NI-LGV-FGSM ------"
#  METHOD="NI-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1 \
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
#  echo "------ NI-Bayesian-FGSM ------"
#  METHOD="NI-Bayesian-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 10 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1 \
#  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
#  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=50)"
#
#  echo "------ NI-FGSM ------"
#  METHOD="NI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ NI-VT-FGSM ------"
#  METHOD="NI-VT-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning|momentum" --n-var-sample 5 --decay-factor 1
#
#  echo "------ NI-LinBP-FGSM ------"
#  METHOD="NI-LinBP-FGSM"
#    $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ NI-SGM-FGSM ------"
#  METHOD="NI-SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2) "\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ NI-FIA-FGSM ------"
#  METHOD="NI-FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1
#
#  echo "------ NI-NAA-FGSM ------"
#  METHOD="NI-NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ NI-DRA-FGSM ------"
#  METHOD="NI-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ NI-GhostNet-FGSM ------"
#  METHOD="NI-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --ghost-attack --decay-factor 1
#
#  echo "------ NI-LGV-FGSM ------"
#  METHOD="NI-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1 \
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
#  echo "------ NI-Bayesian-FGSM ------"
#  METHOD="NI-Bayesian-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "look_ahead_NI" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"  --decay-factor 1 \
#  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
#  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=50)"
#
#  echo "------ VT-FGSM ------"
#  METHOD="VT-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning" --n-var-sample 5
#
#  echo "------ VT-RAP-FGSM ------"
#  METHOD="VT-RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning" --n-var-sample 5
#
#  echo "------ VT-LinBP-FGSM ------"
#  METHOD="VT-LinBP-FGSM"
#    $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "var_tuning" --n-var-sample 5
#
#  echo "------ VT-SGM-FGSM ------"
#  METHOD="VT-SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2) "\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning" --n-var-sample 5
#
#  echo "------ VT-FIA-FGSM ------"
#  METHOD="VT-FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 50 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning" --n-var-sample 5
#
#  echo "------ VT-NAA-FGSM ------"
#  METHOD="VT-NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning" --n-var-sample 5
#
#  echo "------ VT-DRA-FGSM ------"
#  METHOD="VT-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning" --n-var-sample 5
#
#  echo "------ VT-GhostNet-FGSM ------"
#  METHOD="VT-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning" --ghost-attack --n-var-sample 5
#
#  echo "------ VT-LGV-FGSM ------"
#  METHOD="VT-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning" --n-var-sample 5 \
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
  echo "------ VT-Bayesian-FGSM ------"
  METHOD="VT-Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general" \
  --backpropagation "nonlinear" --update-dir-calculation "var_tuning"  --n-var-sample 5 \
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ RAP-FGSM ------"
#  METHOD="RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ RAP-LinBP-FGSM ------"
#  METHOD="RAP-LinBP-FGSM"
#    $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "sgd"
#
#  echo "------ RAP-SGM-FGSM ------"
#  METHOD="RAP-SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma='0.2') "\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ RAP-FIA-FGSM ------"
#  METHOD="RAP-FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ RAP-NAA-FGSM ------"
#  METHOD="RAP-NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ RAP-DRA-FGSM ------"
#  METHOD="RAP-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ RAP-GhostNet-EGSM ------"
#  METHOD="RAP-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ RAP-LGV-FGSM ------"
#  METHOD="RAP-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
  echo "------ RAP-Bayesian-FGSM ------"
  METHOD="RAP-Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 400 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
  --input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
  --backpropagation "nonlinear" --update-dir-calculation "sgd"\
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ LinBP-FGSM ------"
#  METHOD="LinBP-FGSM"
#    $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "sgd"
#
#  echo "------ LinBP-SGM-FGSM ------"
#  METHOD="LinBP-SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "skip_gradient(gamma=0.2) "\
#  --backpropagation "linear" --update-dir-calculation "sgd"
#
#  echo "------ LinBP-DRA-FGSM ------"
#  METHOD="LinBP-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "sgd"
#
#  echo "------ LinBP-GhostNet-EGSM ------"
#  METHOD="LinBP-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ LinBP-LGV-FGSM ------"
#  METHOD="LinBP-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
#  --backpropagation "linear" --update-dir-calculation "sgd" --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
  echo "------ LinBP-Bayesian-FGSM ------"
  METHOD="LinBP-Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
  --input-transformation "" --loss-function "linbp(linbp_layer='3_1')" --grad-calculation "general"\
  --backpropagation "linear" --update-dir-calculation "sgd"\
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ SGM-FGSM ------"
#  METHOD="SGM-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma='0.2')"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SGM-FIA-FGSM ------"
#  METHOD="SGM-FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "skip_gradient(gamma=0.2)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SGM-NAA-FGSM ------"
#  METHOD="SGM-NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "skip_gradient(gamma='0.2')"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SGM-DRA-FGSM ------"
#  METHOD="SGM-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ SGM-GhostNet-EGSM ------"
#  METHOD="SGM-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma='0.2')"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ SGM-LGV-FGSM ------"
#  METHOD="SGM-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
  echo "------ SGM-Bayesian-FGSM ------"
  METHOD="SGM-Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma=0.2)"\
  --backpropagation "nonlinear" --update-dir-calculation "sgd"\
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ FIA-FGSM ------"
#  METHOD="FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ FIA-DRA-FGSM ------"
#  METHOD="FIA-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ FIA-GhostNet-FGSM ------"
#  METHOD="FIA-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ FIA-LGV ------"
#  METHOD="FIA-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
#  echo "------ FIA-Bayesian-FGSM ------"
#  METHOD="FIA-Bayesian-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "fia_loss(fia_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
#  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
#  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=50)"
#
#  echo "------ NAA-FGSM ------"
#  METHOD="NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ NAA-DRA-FGSM ------"
#  METHOD="NAA-DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ NAA-GhostNet-FGSM ------"
#  METHOD="NAA-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ NAA-LGV ------"
#  METHOD="NAA-LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
  echo "------ NAA-Bayesian-FGSM ------"
  METHOD="NAA-Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size 100 --shuffle --n-ensemble 1 \
  --input-transformation "" --loss-function "naa_loss(naa_layer='2')" --grad-calculation "general"\
  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ DRA-FGSM ------"
#  METHOD="DRA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ DRA-GhostNet-FGSM ------"
#  METHOD="DRA-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ GhostNet-FGSM ------"
#  METHOD="GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ LGV ------"
#  METHOD="LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
#  echo "------ LGV-GhostNet ------"
#  METHOD="LGV-GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general" \
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
#  --source-model-refinement "stochastic_weight_collecting(collect=False)" --ghost-attack
#
  echo "------ Bayesian-FGSM ------"
  METHOD="Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
  echo "------ Bayesian-GhostNet-FGSM ------"
  METHOD="Bayesian-GhostNet-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
   --n-iter 100 --batch-size ${BATCH_SIZE} --shuffle --n-ensemble 1 \
  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"  --ghost-attack
#
#  echo "------ ens_logit_I-FGSM ------"
#  METHOD="ens_logit_I-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 100  --batch-size 200 --shuffle --n-ensemble 5 --emsemble-type "logit" \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ ens_logit_TI-FGSM ------"
#  METHOD="ens_logit_TI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 100  --batch-size 100 --shuffle --n-ensemble 5 --emsemble-type "logit"\
#  --input-transformation "TI" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ ens_logit_DI-FGSM ------"
#  METHOD="ens_logit_DI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 100  --batch-size 100 --shuffle --n-ensemble 5 --emsemble-type "logit"\
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ ens_logit_SI-FGSM ------"
#  METHOD="ens_logit_SI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 100  --batch-size 50 --shuffle --n-ensemble 5 --emsemble-type "logit"\
#  --input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ ens_logit_ADMIX-FGSM ------"
#  METHOD="ens_logit_ADMIX-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 100  --batch-size 50 --shuffle --n-ensemble 5 --emsemble-type "logit"\
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ ens_logit_SIA-FGSM ------"
#  METHOD="ens_logit_SIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 100  --batch-size 25 --shuffle --n-ensemble 5 --emsemble-type "logit"\
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ ens_logit_MI-FGSM ------"
#  METHOD="ens_logit_MI-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 100  --batch-size 100 --shuffle --n-ensemble 5 --emsemble-type "logit"\
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#  echo "------ ens_logit_VT-FGSM ------"
#  METHOD="ens_logit_VT-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 100  --batch-size 100 --shuffle --n-ensemble 5 --emsemble-type "logit"\
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "var_tuning"  --n-var-sample 5
#
#  echo "------ ens_logit_RAP-FGSM ------"
#  METHOD="ens_logit_RAP-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 400  --batch-size 100 --shuffle --n-ensemble 5 --emsemble-type "logit"\
#  --input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
  echo "------ ens_logit_SGM-FGSM ------"
  METHOD="ens_logit_SGM-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $ENSEMBLE_PATH_SGM --target-model-path $TARGET_PATH $ARGS_COMMON \
  --n-iter 100  --batch-size 50 --n-ensemble 1 --emsemble-type "logit"\
  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "skip_gradient(gamma='0.2|0.5|0.6')"\
  --backpropagation "nonlinear" --update-dir-calculation "sgd"
##
#  echo "------ ens_logit_FIA-FGSM ------"
#  METHOD="ens_logit_FIA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 100  --batch-size 25 --n-ensemble 5 \
#  --input-transformation "" --loss-function "fia_loss(fia_layer='2|9_0|5|9|3')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ ens_logit_NAA-FGSM ------"
#  METHOD="ens_logit_NAA-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 100  --batch-size 25 --n-ensemble 5 \
#  --input-transformation "" --loss-function "naa_loss(naa_layer='2|9_0|5|9|3')" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ ens_logit_GhostNet-FGSM ------"
#  METHOD="ens_logit_GhostNet-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 100  --batch-size 100 --shuffle --n-ensemble 5 --emsemble-type "logit"\
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ ens_logit_LGV-FGSM ------"
#  METHOD="ens_logit_LGV-FGSM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 100  --batch-size 100 --shuffle --n-ensemble 5 --emsemble-type "logit"\
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
#  --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
  echo "------ ens_logit_Bayesian-FGSM ------"
  METHOD="ens_logit_Bayesian-FGSM"
  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
  --n-iter 100  --batch-size 100 --shuffle --n-ensemble 5 --emsemble-type "logit"\
  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "general"\
  --backpropagation "nonlinear" --update-dir-calculation "sgd" \
  --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
  --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=100)"
#
#  echo "------ CWA ------"
#  METHOD="CWA"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 10  --batch-size 40 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ CWA-DI ------"
#  METHOD="CWA-DI"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 10  --batch-size 40 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#  --input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ CWA-SI ------"
#  METHOD="CWA-SI"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 10  --batch-size 25 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#  --input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ CWA-ADMIX ------"
#  METHOD="CWA-ADMIX"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 10  --batch-size 25 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#  --input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ CWA-SIA ------"
#  METHOD="CWA-SIA"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 10  --batch-size 25 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#  --input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ CWA-MI ------"
#  METHOD="CWA-MI"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 10  --batch-size 40 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)"\
#  --backpropagation "nonlinear" --update-dir-calculation "momentum"
#
#  echo "------ CWA-VT ------"
#  METHOD="CWA-VT"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 10  --batch-size 50 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --n-var-sample 5
#
#  echo "------ CWA-GhostNet ------"
#  METHOD="CWA-GhostNet"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 10  --batch-size 50 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#  echo "------ CWA-NAA ------"
#  METHOD="CWA-NAA"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 10  --batch-size 10 --n-ensemble 5 --decay-factor 1 \
#  --input-transformation "" --loss-function "naa_loss(naa_layer='2|9_0|5|9|3')" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#  echo "------ CWA-SGM ------"
#  METHOD="CWA-SGM"
#  $ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" --csv-export-path $PATH_CSV \
#  --source-model-path $ENSEMBLE_PATH_SGM --target-model-path $TARGET_PATH $ARGS_COMMON \
#  --n-iter 10  --batch-size 50 --n-ensemble 3 --emsemble-type "logit" --decay-factor 1 \
#  --input-transformation "" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)"\
#  --backpropagation "nonlinear" --update-dir-calculation "sgd"
done
