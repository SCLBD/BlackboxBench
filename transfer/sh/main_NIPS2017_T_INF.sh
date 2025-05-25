#!/bin/bash -l

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "------------------------------------------"
echo "Craft adversarial examples by all method"
echo "Experiment setting"
echo "Dataset: NIPS2017"
echo "Attack: targeted inf"
echo "Surrogate(single): resnet50 vgg19_bn densenet121 inc-v3 vit"
echo "------------------------------------------"

ATTACK="python -u main_attack.py"

SOURCES=(resnet50)  #TODO: SOURCES=(resnet50 vgg19_bn densenet121 inc_v3 vit)
ENSEMBLE_PATH="NIPS2017/pretrained/resnet50 NIPS2017/pretrained/vgg19_bn NIPS2017/pretrained/densenet121
               NIPS2017/pretrained/inception_v3 NIPS2017/pretrained/vit_b_16"
TARGET_PATH="NIPS2017/pretrained/resnet50 NIPS2017/pretrained/vgg19_bn NIPS2017/pretrained/resnet152
        NIPS2017/pretrained/inception_v3 NIPS2017/pretrained/densenet121 NIPS2017/pretrained/mobilenet_v2
        NIPS2017/pretrained/senet154 NIPS2017/pretrained/resnext101 NIPS2017/pretrained/wrn101
        NIPS2017/pretrained/pnasnet5_l NIPS2017/pretrained/mnasnet NIPS2017/pretrained/vit_b_16
        NIPS2017/pretrained/swin_b NIPS2017/pretrained/convnext_b NIPS2017/pretrained/adv_resnet50_Salman2020Do_R50
        NIPS2017/pretrained/adv_swin_b_Liu2023Comprehensive_Swin_B NIPS2017/pretrained/adv_convnext_b_Liu2023Comprehensive_ConvNeXt_B"

ARGS_COMMON="--epsilon 0.06 --norm-step 0.006 --seed 0 --targeted"  # targeted attack
PATH_CSV="./csv_files/NIPS2017_T_INF.csv"
PATH_ADV_BASE="./adv_imgs/NIPS2017/targeted/l_inf"
PATH_JSON_BASE="./json_files/NIPS2017/targeted/l_inf"

echo "-----------ENSEMBLE-----------"
BATCH_SIZE=100
#echo "------ ens_logit_I-FGSM ------"
#METHOD="ens_logit_I-FGSM"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/ensemble/${METHOD}" \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ ens_loss_I-FGSM ------"
#METHOD="ens_loss_I-FGSM"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/ensemble/${METHOD}" \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ ens_longitudinal_I-FGSM ------"
#METHOD="ens_longitudinal_I-FGSM"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/ensemble/${METHOD}" \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
echo "------ AdaEA ------"
METHOD="AdaEA"
$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/ensemble/${METHOD}" \
--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
##echo "------ CWA ------"
##METHOD="CWA"
##$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 10 --save-dir "${PATH_ADV_BASE}/ensemble/${METHOD}" \
##--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size 40

#echo "------ CWA ------"
#METHOD="CWA"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/ensemble/${METHOD}" \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size 40

##echo "------ CWA ------"
##METHOD="CWA_2"
##$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 10 --save-dir "${PATH_ADV_BASE}/ensemble/${METHOD}" \
##--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size 40
##
##echo "------ CWA ------"
##METHOD="CWA_2"
##$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/ensemble/${METHOD}" \
##--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size 40
##
##echo "------ CWA ------"
##METHOD="CWA_3"
##$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 10 --save-dir "${PATH_ADV_BASE}/ensemble/${METHOD}" \
##--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size 40
##
##echo "------ CWA ------"
##METHOD="CWA_3"
##$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/ensemble/${METHOD}" \
##--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size 40
#
#echo "------ SVRE ------"
#METHOD="SVRE"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/ensemble/${METHOD}" \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}


for SOURCE in "${SOURCES[@]}" ; do
  SOURCE_PATH="NIPS2017/pretrained/$SOURCE"

  if [ $SOURCE = 'resnet50' ]; then
    BATCH_SIZE=100
  elif [ $SOURCE = 'densenet121' ]; then
    BATCH_SIZE=300
  elif [ $SOURCE = 'inception_v3' ]; then
    BATCH_SIZE=300
  elif [ $SOURCE = 'vgg19_bn' ]; then
    BATCH_SIZE=300
  elif [ $SOURCE = 'vit_b_16' ]; then
    BATCH_SIZE=300
  fi

  echo "------ I-FGSM ------"
#  METHOD="I-FGSM"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ DI2-FGSM ------"
#  METHOD="DI2-FGSM"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ MI-FGSM ------"
#  METHOD="MI-FGSM"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ TI-FGSM ------"
#  METHOD="TI-FGSM"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ MI-DI2-TI-FGSM ------"
#  METHOD="MI-DI2-TI-FGSM"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ random_start ------"
#  METHOD="random_start"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ NI ------"
#  METHOD="NI"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ PI ------"
#  METHOD="PI"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ VT ------"
#  METHOD="VT"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ RAP ------"
#  METHOD="RAP"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 400 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ SI ------"
#  METHOD="SI"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ admix ------"
#  METHOD="admix"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ MI-DI ------"
#  METHOD="MI-DI"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ MI-DI-TI-SI ------"
#  METHOD="MI-DI-TI-SI"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ VMI ------"
#  METHOD="VMI"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ VNI ------"
#  METHOD="VNI"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ SI-RAP ------"
#  METHOD="SI-RAP"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 400 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#  echo "------ SIA ------"
#  METHOD="SIA"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

#  echo "------ PGN ------"
#  METHOD="PGN"
#  $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#  --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

  if [ $SOURCE = 'resnet50' ]; then
    echo "------ Styless ------"
#    METHOD="Styless"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#    echo "------ RD ------"
#    METHOD="RD"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/RD/resnet50/0.005" --source-model-refinement "sample_from_isotropic(std=0.005, n_models=10)"
#
#    echo "------ LinBP ------"
#    METHOD="LinBP"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --batch-size 100   --loss-function "linbp(linbp_layer='3_1')"
#
#    echo "------ SGM ------"
#    METHOD="SGM"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --batch-size ${BATCH_SIZE}   --grad-calculation "skip_gradient(gamma=0.2)"
#
#    echo "------ ILA ------"
#    METHOD="ILA_BSL"
#    BSL_ADV="${PATH_ADV_BASE}/$SOURCE/bsl_adv"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 30 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --bsl-adv-img-path $BSL_ADV --batch-size ${BATCH_SIZE}
#    METHOD="ILA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --bsl-adv-img-path $BSL_ADV --batch-size 100 --loss-function "ila_loss(ila_layer='3_4')"
#
#    echo "------ GhostNet ------"
#    METHOD="GhostNet"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#    echo "------ DRA ------"
#    METHOD="DRA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV  --batch-size ${BATCH_SIZE}
#
#    echo "------ IAA ------"
#    METHOD="IAA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path "NIPS2017/IAA/resnet50" --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV  --batch-size ${BATCH_SIZE}
#
#    echo "------ LGV ------"
#    METHOD="LGV"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
#    $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
#    echo "------ SWA ------"
#    METHOD="SWA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/SWA/$SOURCE/swa_model" --source-model-refinement "stochastic_weight_averaging(collect=False)"
#
#    echo "------ Bayesian_attack ------"
#    METHOD="Bayesian_attack"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
#    --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=50)"
#
#    echo "------ LGV-GhostNet ------"
#    METHOD="LGV-GhostNet"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
#    $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
#    echo "------ no FIA ------"
#
#    echo "------ no NAA ------"



  elif [ $SOURCE = 'densenet121' ]; then
    echo "------ RD ------"
#    METHOD="RD"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/RD/densenet121/0.01" --source-model-refinement "sample_from_isotropic(std=0.01, n_models=10)"
#
#    echo "------ LinBP TBD ------"
#
#    echo "------ SGM ------"
#    METHOD="SGM"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --batch-size ${BATCH_SIZE}   --grad-calculation "skip_gradient(gamma=0.5)"
#
#    echo "------ ILA ------"
#    METHOD="ILA_BSL"
#    BSL_ADV="${PATH_ADV_BASE}/$SOURCE/bsl_adv"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 30 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --bsl-adv-img-path $BSL_ADV --batch-size ${BATCH_SIZE}
#    METHOD="ILA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --bsl-adv-img-path $BSL_ADV --batch-size 100 --loss-function "ila_loss(ila_layer='5')"
#
#    echo "------ GhostNet ------"
#    METHOD="GhostNet"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#    echo "------ no FIA ------"
#
#    echo "------ no NAA ------"
#
#    echo "------ DRA ------"
#    METHOD="DRA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path "NIPS2017/DRA/DRA_densenet121.pth" --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV  --batch-size ${BATCH_SIZE}
#
#    echo "------ LGV ------"
#    METHOD="LGV"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
#    $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
#    echo "------ SWA ------"
#    METHOD="SWA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/SWA/$SOURCE/swa_model" --source-model-refinement "stochastic_weight_averaging(collect=False)"
#
#    echo "------ Bayesian_attack ------"
#    METHOD="Bayesian_attack"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
#    --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=50)"
#
#    echo "------ LGV-GhostNet ------"
#    METHOD="LGV-GhostNet"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
#    $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --source-model-refinement "stochastic_weight_collecting(collect=False)"



  elif [ $SOURCE = 'inception_v3' ]; then
    echo "------ RD ------"
#    METHOD="RD"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/RD/inception_v3/0.01" --source-model-refinement "sample_from_isotropic(std=0.01, n_models=10)"
#
#    echo "------ LinBP TBD ------"
#
#    echo "------ no SGM ------"
#
#    echo "------ ILA ------"
#    METHOD="ILA_BSL"
#    BSL_ADV="${PATH_ADV_BASE}/$SOURCE/bsl_adv"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 30 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --bsl-adv-img-path $BSL_ADV --batch-size ${BATCH_SIZE}
#    METHOD="ILA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --bsl-adv-img-path $BSL_ADV --batch-size 50 --loss-function "ila_loss(ila_layer='9')"
#
#    echo "------ no FIA ------"
#
#    echo "------ no NAA ------"
#
#    echo "------ GhostNet ------"
#    METHOD="GhostNet"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#    echo "------ LGV ------"
#    METHOD="LGV"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
#    $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
#    echo "------ SWA ------"
#    METHOD="SWA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/SWA/$SOURCE/swa_model" --source-model-refinement "stochastic_weight_averaging(collect=False)"
#
#    echo "------ Bayesian_attack ------"
#    METHOD="Bayesian_attack"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
#    --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=50)"
#
#    echo "------ LGV-GhostNet ------"
#    METHOD="LGV-GhostNet"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
#    $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --source-model-refinement "stochastic_weight_collecting(collect=False)"



  elif [ $SOURCE = 'vgg19_bn' ]; then
    echo "------ RD ------"
#    METHOD="RD"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/RD/vgg19_bn/0.01" --source-model-refinement "sample_from_isotropic(std=0.01, n_models=10)"
#
#    echo "------ LinBP ------"
#    METHOD="LinBP"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --batch-size 50   --loss-function "linbp(linbp_layer=46)"
#
#    echo "------ no SGM ------"
#
#    echo "------ ILA ------"
#    METHOD="ILA_BSL"
#    BSL_ADV="${PATH_ADV_BASE}/$SOURCE/bsl_adv"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 30 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --bsl-adv-img-path $BSL_ADV --batch-size ${BATCH_SIZE}
#    METHOD="ILA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --bsl-adv-img-path $BSL_ADV --batch-size 100 --loss-function "ila_loss(ila_layer='9_0')"
#
#    echo "------ no FIA ------"
#
#    echo "------ no NAA ------"
#
#    echo "------ GhostNet ------"
#    METHOD="GhostNet"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#    echo "------ DRA ------"
#    METHOD="DRA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path "NIPS2017/DRA/DRA_vgg19_bn.pth" --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV  --batch-size ${BATCH_SIZE}
#
#    echo "------ LGV ------"
#    METHOD="LGV"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
#    $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
#    echo "------ SWA ------"
#    METHOD="SWA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/SWA/$SOURCE/swa_model" --source-model-refinement "stochastic_weight_averaging(collect=False)"
#
#    echo "------ Bayesian_attack ------"
#    METHOD="Bayesian_attack"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
#    --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=50)"
#
#    echo "------ LGV-GhostNet ------"
#    METHOD="LGV-GhostNet"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
#    $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --source-model-refinement "stochastic_weight_collecting(collect=False)"



  elif [ $SOURCE = 'vit_b_16' ]; then
    echo "------ RD ------"
#    METHOD="RD"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/RD/vit_b_16/0.005" --source-model-refinement "sample_from_isotropic(std=0.005, n_models=10)"
#
#    echo "------ GhostNet ------"
#    METHOD="GhostNet"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#    echo "------ ILA ------"
#    METHOD="ILA_BSL"
#    BSL_ADV="${PATH_ADV_BASE}/$SOURCE/bsl_adv"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 30 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --bsl-adv-img-path $BSL_ADV --batch-size ${BATCH_SIZE}
#    METHOD="ILA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --bsl-adv-img-path $BSL_ADV --batch-size 50 --loss-function "ila_loss(ila_layer='3')"
#
#    echo "------ SGM ------"
#    METHOD="SGM"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --batch-size ${BATCH_SIZE}   --grad-calculation "skip_gradient(gamma=0.6)"
#
#    echo "------ LGV ------"
#    METHOD="LGV"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
#    $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --source-model-refinement "stochastic_weight_collecting(collect=False)"
#
#    echo "------ SWA ------"
#    METHOD="SWA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/SWA/$SOURCE/swa_model" --source-model-refinement "stochastic_weight_averaging(collect=False)"
#
#    echo "------ Bayesian_attack ------"
#    METHOD="Bayesian_attack"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
#    --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=50)"
#
#    echo "------ LGV-GhostNet ------"
#    METHOD="LGV-GhostNet"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
#    $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#    --source-model-refinement "stochastic_weight_collecting(collect=False)"


  fi


done