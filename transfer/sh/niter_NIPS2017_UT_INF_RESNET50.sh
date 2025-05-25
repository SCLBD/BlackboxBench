#!/bin/bash -l

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "------------------------------------------"
echo "Craft adversarial examples by all methods with different numbers of iteration"
echo "Experiment setting"
echo "Dataset: NIPS2017"
echo "Attack: untargeted inf epsilon(0.03)"
echo "Surrogate(single): resnet50"
echo "Numbers of iteration: 10 50 100 200 300 400"
echo "------------------------------------------"

ATTACK="python -u main_attack.py"

SOURCES=(resnet50)
ENSEMBLE_PATH="NIPS2017/pretrained/resnet50 NIPS2017/pretrained/vgg19_bn NIPS2017/pretrained/densenet121
               NIPS2017/pretrained/inception_v3 NIPS2017/pretrained/vit_b_16"
TARGET_PATH="NIPS2017/pretrained/resnet50 NIPS2017/pretrained/vgg19_bn NIPS2017/pretrained/resnet152
        NIPS2017/pretrained/inception_v3 NIPS2017/pretrained/densenet121 NIPS2017/pretrained/mobilenet_v2
        NIPS2017/pretrained/senet154 NIPS2017/pretrained/resnext101 NIPS2017/pretrained/wrn101
        NIPS2017/pretrained/pnasnet5_l NIPS2017/pretrained/mnasnet NIPS2017/pretrained/vit_b_16
        NIPS2017/pretrained/swin_b NIPS2017/pretrained/convnext_b NIPS2017/pretrained/adv_resnet50_Salman2020Do_R50
        NIPS2017/pretrained/adv_swin_b_Liu2023Comprehensive_Swin_B NIPS2017/pretrained/adv_convnext_b_Liu2023Comprehensive_ConvNeXt_B"

PATH_CSV="./csv_files/niter_NIPS2017_UT_INF_RESNET50.csv"
PATH_JSON_BASE="./json_files/NIPS2017/untargeted/l_inf"
NITERS=(10 50 100 200 300 400)



for SOURCE in "${SOURCES[@]}" ; do
  SOURCE_PATH="NIPS2017/pretrained/$SOURCE"

  for NITER in "${NITERS[@]}" ; do
    PATH_ADV_BASE="./adv_imgs/analysis/niter_NIPS2017_UT_INF_RESNET50/$NITER"
    ARGS_COMMON="--epsilon 0.03 --norm-step 0.00392157 --seed 0 --n-iter $NITER"

#    echo "------ ens_logit_I-FGSM ------"
#    METHOD="ens_logit_I-FGSM"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" \
#    --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --batch-size 100
#
#    echo "------ ens_loss_I-FGSM ------"
#    METHOD="ens_loss_I-FGSM"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" \
#    --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --batch-size 100
#
#    echo "------ ens_longitudinal_I-FGSM ------"
#    METHOD="ens_longitudinal_I-FGSM"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" \
#    --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --batch-size 100
#
#    echo "------ CWA ------"
#    METHOD="CWA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" \
#    --source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --batch-size 50

echo "------ AdaEA ------"
METHOD="AdaEA"
$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" \
--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
--batch-size 200

#
#    echo "------ SIA ------"
#    METHOD="SIA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --batch-size 100
#
#    echo "------ PGN ------"
#    METHOD="PGN"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --batch-size 100

#    echo "------ Styless ------"
#    METHOD="Styless"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --batch-size 100
#
#    echo "------ I-FGSM ------"
#    METHOD="I-FGSM"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ random_start ------"
#    METHOD="random_start"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ MI-FGSM ------"
#    METHOD="MI-FGSM"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ NI ------"
#    METHOD="NI"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ PI ------"
#    METHOD="PI"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ VT ------"
#    METHOD="VT"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ RAP ------"
#    METHOD="RAP"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --input-transformation "add_reverse_perturbation(late_start=0)"
#
#    echo "------ LinBP ------"
#    METHOD="LinBP"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ SGM ------"
#    METHOD="SGM"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ DI2-FGSM ------"
#    METHOD="DI2-FGSM"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --input-transformation "DI(in_size=224, out_size=256)"
#
#    echo "------ SI ------"
#    METHOD="SI"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ admix ------"
#    METHOD="admix"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ TI ------"
#    METHOD="TI"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ MI-DI ------"
#    METHOD="MI-DI"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --input-transformation "DI(in_size=224, out_size=256)"
#
#    echo "------ MI-DI-TI ------"
#    METHOD="MI-DI-TI"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --input-transformation "DI(in_size=224, out_size=256)"
#
#    echo "------ MI-DI-TI-SI ------"
#    METHOD="MI-DI-TI-SI"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --input-transformation "DI(in_size=224, out_size=256)"
#
#    echo "------ FIA ------"
#    METHOD="FIA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ NAA ------"
#    METHOD="NAA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ DRA ------"
#    METHOD="DRA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ RD ------"
#    METHOD="RD"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ GhostNet ------"
#    METHOD="GhostNet"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ IAA ------"
#    METHOD="IAA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path "NIPS2017/IAA/resnet50" --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ ILA ------"
#    METHOD_BSL="ILA_BSL"
#    BSL_ADV="${PATH_ADV_BASE}/$SOURCE/bsl_adv"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD_BSL}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD_BSL}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH --epsilon 0.03 --norm-step 0.00392157 --seed 0 --n-iter 10 --csv-export-path $PATH_CSV \
#    --bsl-adv-img-path $BSL_ADV
#    METHOD="ILA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --bsl-adv-img-path $BSL_ADV
#
#    echo "------ LGV ------"
#    METHOD="LGV"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path "NIPS2017/LGV/models/ImageNet/resnet50/cSGD/seed0/original" --target-model-path $TARGET_PATH \
#    $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ SWA ------"
#    METHOD="SWA"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ Bayesian_attack ------"
#    METHOD="Bayesian_attack"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ VMI ------"
#    METHOD="VMI"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ VNI ------"
#    METHOD="VNI"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
##    echo "------ admix-RAP ------"
##    METHOD="admix-RAP"
##    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
##    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#    echo "------ SI-RAP ------"
#    METHOD="SI-RAP"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#    --input-transformation "add_reverse_perturbation(late_start=0)|SI(n_copies=5)"
#
#    echo "------ LGV-GhostNet ------"
#    METHOD="LGV-GhostNet"
#    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#    --source-model-path "NIPS2017/LGV/models/ImageNet/resnet50/cSGD/seed0/original/ImageNet-ResNet50-052e7f78e4db--1564492444-1.pth.tar" \
#    --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV

  done

done