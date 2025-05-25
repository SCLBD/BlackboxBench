#!/bin/bash -l

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "------------------------------------------"
echo "Craft adversarial examples by all method with different norm of perturbation"
echo "6 hours for all methods without RAP+admix (cost about 3 hours)"
echo "Experiment setting"
echo "Dataset: NIPS2017"
echo "Attack: untargeted inf optimal_niter"
echo "Surrogate(single): resnet50"
echo "Norm of perturbation: 0.03"
echo "------------------------------------------"

ATTACK="python -u main_attack.py"

SOURCES=(resnet50)
TARGET_PATH="NIPS2017/pretrained/resnet50
             NIPS2017/pretrained/vgg19_bn
             NIPS2017/pretrained/resnet152
             NIPS2017/pretrained/inception_v3"

PATH_JSON_BASE="./json_files/NIPS2017/untargeted/l_inf"
PATH_CSV="./csv_files/visualization_fullgrad_NIPS2017_UT_INF_RESNET50.csv"
EPSILONS=(0.03)
NORM_STEPS=(0.00392157)



for SOURCE in "${SOURCES[@]}" ; do
  SOURCE_PATH="NIPS2017/pretrained/$SOURCE"

  if [ $SOURCE = 'resnet50' ]; then
    BATCH_SIZE=200
  elif [ $SOURCE = 'densenet121' ]; then
    BATCH_SIZE=300
  elif [ $SOURCE = 'inception_v3' ]; then
    BATCH_SIZE=300
  elif [ $SOURCE = 'vgg19_bn' ]; then
    BATCH_SIZE=300
  elif [ $SOURCE = 'vit_b_16' ]; then
    BATCH_SIZE=300
  fi

  for i in {0..1} ; do
    PATH_ADV_BASE="./adv_imgs/analysis/visualization_fullgrad_NIPS2017_UT_INF_RESNET50"
    ARGS_COMMON="--epsilon ${EPSILONS[i]} --norm-step ${NORM_STEPS[i]} --seed 0"

    echo "------ I-FGSM ------"
    METHOD="I-FGSM"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

    echo "------ random_start ------"
    METHOD="random_start"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

    echo "------ MI-FGSM ------"
    METHOD="MI-FGSM"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 10 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

    echo "------ NI ------"
    METHOD="NI"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 10 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

    echo "------ PI ------"
    METHOD="PI"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 10 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

    echo "------ VT ------"
    METHOD="VT"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

    echo "------ RAP ------"
    METHOD="RAP"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 400 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

    echo "------ DI2-FGSM ------"
    METHOD="DI2-FGSM"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

    echo "------ SI ------"
    METHOD="SI"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

    echo "------ admix ------"
    METHOD="admix"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

    echo "------ TI ------"
    METHOD="TI"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

    echo "------ RD ------"
    METHOD="RD"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
    --rfmodel-dir "NIPS2017/RD/resnet50/0.005" --source-model-refinement "sample_from_isotropic(std=0.005, n_models=10)"

    echo "------ GhostNet ------"
    METHOD="GhostNet"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}

    echo "------ LinBP ------"
    METHOD="LinBP"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
    --loss-function "linbp(linbp_layer='3_1')" --batch-size 100

    echo "------ SGM ------"
    METHOD="SGM"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
    --grad-calculation "skip_gradient(gamma=0.2)" --batch-size ${BATCH_SIZE}

    echo "------ ILA ------"
    METHOD_BSL="ILA_BSL"
    BSL_ADV="${PATH_ADV_BASE}/$SOURCE/bsl_adv"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD_BSL}.json" --n-iter 10 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD_BSL}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
    --bsl-adv-img-path $BSL_ADV --batch-size ${BATCH_SIZE}
    METHOD="ILA"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
    --bsl-adv-img-path $BSL_ADV --batch-size 100 --loss-function "ila_loss(ila_layer='2')"

    echo "------ FIA ------"
    METHOD="FIA"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
    --loss-function "fia_loss(fia_layer='2')" --batch-size 100

    echo "------ NAA ------"
    METHOD="NAA"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
    --loss-function "naa_loss(naa_layer='2')" --batch-size 100

    echo "------ DRA ------"
    METHOD="DRA"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
    --batch-size ${BATCH_SIZE}

    echo "------ IAA ------"
    METHOD="IAA"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path "NIPS2017/IAA/resnet50" --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
    --batch-size ${BATCH_SIZE}

    echo "------ LGV ------"
    METHOD="LGV"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
    $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
    --source-model-refinement "stochastic_weight_collecting(collect=False)"

    echo "------ SWA ------"
    METHOD="SWA"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
    --rfmodel-dir "NIPS2017/SWA/$SOURCE/swa_model" --source-model-refinement "stochastic_weight_averaging(collect=False)"

    echo "------ Bayesian_attack ------"
    METHOD="Bayesian_attack"
    $ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}/final" \
    --source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
    --rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
    --source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=50)"

  done

done