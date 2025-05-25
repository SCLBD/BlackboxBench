#!/bin/bash -l

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "------------------------------------------"
echo "Main results for new RAP paper"
echo "Craft adversarial examples by all method"
echo "Experiment setting"
echo "Dataset: NIPS2017"
echo "Attack: untargeted inf"
echo "Surrogate(ensemble): resnet50 convnext vit xcit(at) resnet50(at)"
echo "------------------------------------------"

ATTACK="python -u main_attack.py"

ENSEMBLE_PATH="NIPS2017/pretrained/adv_resnet50_gelu
            NIPS2017/pretrained/adv_xcit_s
            NIPS2017/pretrained/convnext_t_tv
            NIPS2017/pretrained/vit_b_16_google
            NIPS2017/pretrained/resnet50"
TARGET_PATH="NIPS2017/pretrained/alexnet
            NIPS2017/pretrained/vgg16_bn
            NIPS2017/pretrained/densenet201
            NIPS2017/pretrained/googlenet
            NIPS2017/pretrained/shufflenet_v2_x1_0
            NIPS2017/pretrained/mobilenet_v2
            NIPS2017/pretrained/mobilenet_v3_large
            NIPS2017/pretrained/mnasnet
            NIPS2017/pretrained/efficientnet_b7
            NIPS2017/pretrained/convnext_l_tv
            NIPS2017/pretrained/vit_s_16
            NIPS2017/pretrained/deit_s
            NIPS2017/pretrained/poolformer_s
            NIPS2017/pretrained/tnt_s
            NIPS2017/pretrained/swin_s
            NIPS2017/pretrained/xcit_s
            NIPS2017/pretrained/cait_s
            NIPS2017/pretrained/adv_rawrn_101_2_Peng2023Robust
            NIPS2017/pretrained/adv_wrn_50_2_Salman2020Do_50_2
            NIPS2017/pretrained/adv_resnet50_Wong2020Fast
            NIPS2017/pretrained/adv_convnext_l_Liu2023Comprehensive_ConvNeXt_L
            NIPS2017/pretrained/adv_convnext_b_Liu2023Comprehensive_ConvNeXt_B
            NIPS2017/pretrained/adv_convnext_l_convstem_Singh2023Revisiting_ConvNeXt_L_ConvStem
            NIPS2017/pretrained/adv_convnext_b_convstem_Singh2023Revisiting_ConvNeXt_B_ConvStem
            NIPS2017/pretrained/tf2torch_ens3_adv_inc_v3
            NIPS2017/pretrained/tf2torch_ens4_adv_inc_v3
            NIPS2017/pretrained/tf2torch_ens_adv_inc_res_v2
            NIPS2017/pretrained/adv_swin_b_Liu2023Comprehensive_Swin_B
            NIPS2017/pretrained/adv_swin_l_Liu2023Comprehensive_Swin_L
            NIPS2017/pretrained/adv_xcit_l_Debenedetti2022Light_XCiT_L12
            NIPS2017/pretrained/adv_vit_b_convstem_Singh2023Revisiting_ViT_B_ConvStem"

ARGS_COMMON="--epsilon 0.015686274509804 --norm-step 0.007843137254902 --seed 0 --norm-type inf" # untargeted linf
PATH_CSV="./new_csv_files/NIPS2017_UT_INF.csv"
PATH_ADV_BASE="./new_adv_imgs/NIPS2017/untargeted/l_inf"
PATH_JSON_BASE="./json_files/NIPS2017/untargeted/l_inf"

BATCH_SIZE=200
N_ITER=200

#echo "------ NEW-RAP ------"
#METHOD="NEW-RAP-noflat"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 200 --n-ensemble 1 \
#--input-transformation "" --loss-function "cross_entropy" --grad-calculation "rap_new(model_num=40, late_start=40, reverse_step=5, reverse_step_size=0.1)" \
#--backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#echo "------ NEW-RAP ------"
#METHOD="NEW-RAP"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 200 --n-ensemble 1 \
#--input-transformation "" --loss-function "cross_entropy" --grad-calculation "rap_new(model_num=40, late_start=5, reverse_step=5, reverse_step_size=0.1)" \
#--backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#echo "------ NEW-RAP-nodiverse-25 ------"
#METHOD="NEW-RAP-nodiverse-25"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter 25  --batch-size 200 --n-ensemble 1 \
#--input-transformation "" --loss-function "cross_entropy" --grad-calculation "rap_new_nodiverse(late_start=0, reverse_step=5, reverse_step_size=0.1)" \
#--backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#echo "------ Flat-RAP-MI-40 ------"
#METHOD="Flat-RAP-MI-40"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter 40  --batch-size 100 --n-ensemble 5 \
#--input-transformation "add_reverse_perturbation(late_start=5)" --loss-function "cross_entropy" --grad-calculation "flat_rap" \
#--backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#echo "------ ens_logit_I-FGSM ------"
#METHOD="ens_logit_I-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 200 --n-ensemble 5 --emsemble-type "logit" \
#--input-transformation "" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"

#echo "------ ens_logit_TI-FGSM ------"
#METHOD="ens_logit_TI-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 100 --n-ensemble 5 --emsemble-type "logit" \
#--input-transformation "TI" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ ens_logit_DI-FGSM ------"
#METHOD="ens_logit_DI-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 100 --n-ensemble 5 --emsemble-type "logit" \
#--input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ ens_logit_SI-FGSM ------"
#METHOD="ens_logit_SI-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 50 --n-ensemble 5 --emsemble-type "logit"\
#--input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ ens_logit_ADMIX-FGSM ------"
#METHOD="ens_logit_ADMIX-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 50 --n-ensemble 5 --emsemble-type "logit"\
#--input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ ens_logit_SIA-FGSM ------"
#METHOD="ens_logit_SIA-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 25 --n-ensemble 5 --emsemble-type "logit" \
#--input-transformation "siA(n_block=3, n_copies=10)" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ ens_logit_MI-FGSM ------"
#METHOD="ens_logit_MI-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 100 --n-ensemble 5 --emsemble-type "logit" \
#--input-transformation "" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#echo "------ ens_logit_MI-FGSM ------"
#METHOD="ens_logit_MI-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter 10  --batch-size 100 --n-ensemble 5 --emsemble-type "logit" \
#--input-transformation "" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1

#echo "------ ens_logit_VT-FGSM ------"
#METHOD="ens_logit_VT-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 100 --n-ensemble 5 --emsemble-type "logit"\
#--input-transformation "" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "var_tuning"  --n-var-sample 5
#
#echo "------ ens_logit_RAP-FGSM ------"
#METHOD="ens_logit_RAP-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter 400  --batch-size 100 --n-ensemble 5 --emsemble-type "logit"\
#--input-transformation "add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ ens_logit_PGN-FGSM ------"
#METHOD="ens_logit_PGN-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 40 --n-ensemble 5 --emsemble-type "logit" \
#--input-transformation "" --loss-function "cross_entropy" --grad-calculation "PGN_grad(zeta=3, delta=0.5, N=20)" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ ens_logit_GhostNet-FGSM ------"
#METHOD="ens_logit_GhostNet-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 100 --n-ensemble 5 --emsemble-type "logit" \
#--input-transformation "" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd" --ghost-attack
#
#echo "------ CWA ------"
#METHOD="CWA-10"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter 10  --batch-size 50 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#--input-transformation "" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ CWA ------"
#METHOD="CWA-200"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 50 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#--input-transformation "" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ Flat-CWA-40 ------"
#METHOD="Flat-CWA-40"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter 40  --batch-size 100 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#--input-transformation "" --loss-function "cross_entropy" --grad-calculation "flat_cwa(reverse_step_size=0.001, inner_step_size=50)" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
# COMPOSITE
#echo "------ NEW-RAP-DI ------"
#METHOD="NEW-RAP-DI"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 250 --n-ensemble 1 \
#--input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "rap_new(model_num=40, late_start=5, reverse_step=5, reverse_step_size=0.1)" \
#--backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#echo "------ NEW-RAP-SI-----"
#METHOD="NEW-RAP-SI"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 250 --n-ensemble 1 \
#--input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "rap_new(model_num=40, late_start=5, reverse_step=5, reverse_step_size=0.1)" \
#--backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#echo "------ NEW-RAP-Admix-----"
#METHOD="NEW-RAP-Admix"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 250 --n-ensemble 1 \
#--input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "rap_new(model_num=40, late_start=5, reverse_step=5, reverse_step_size=0.1)" \
#--backpropagation "nonlinear" --update-dir-calculation "momentum" --decay-factor 1
#
#echo "------ DI-RAP-FGSM ------"
#METHOD="DI-RAP-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
# --n-iter 400 --batch-size ${BATCH_SIZE}  --n-ensemble 5 \
#--input-transformation "add_reverse_perturbation(late_start=100)|DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ SI-RAP-FGSM ------"
#METHOD="SI-RAP-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
# --n-iter 400 --batch-size ${BATCH_SIZE}  --n-ensemble 5 \
#--input-transformation "add_reverse_perturbation(late_start=100)|SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"

#echo "------ SI-RAP-FGSM ------" # seems wrong
#METHOD="SI-RAP-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
# --n-iter 400 --batch-size ${BATCH_SIZE}  --n-ensemble 5 \
#--input-transformation "SI(n_copies=5)|add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ Admix-RAP-FGSM ------"
#METHOD="Admix-RAP-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
# --n-iter 400 --batch-size ${BATCH_SIZE}  --n-ensemble 5 \
#--input-transformation "add_reverse_perturbation(late_start=100)|admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ Admix-RAP-FGSM ------" # seems wrong
#METHOD="Admix-RAP-FGSM"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
# --n-iter 400 --batch-size ${BATCH_SIZE}  --n-ensemble 5 \
#--input-transformation "admix(strength=0.2, n_samples=3)|add_reverse_perturbation(late_start=100)" --loss-function "cross_entropy" --grad-calculation "general" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ CWA-DI ------"
#METHOD="CWA-DI"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 40 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#--input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ CWA-SI ------"
#METHOD="CWA-SI"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 40 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#--input-transformation "SI(n_copies=5)" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ CWA-ADMIX ------"
#METHOD="CWA-ADMIX"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
#--n-iter $N_ITER  --batch-size 40 --n-ensemble 5 --emsemble-type "logit" --decay-factor 1 \
#--input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "cwa_new(reverse_step_size=0.001, inner_step_size=50)" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ PGN-TI ------"
#METHOD="PGN-TI"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
# --n-iter $N_ITER --batch-size 40  --n-ensemble 5 \
#--input-transformation "TI" --loss-function "cross_entropy" --grad-calculation "PGN_grad(zeta=3, delta=0.5, N=20)" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ PGN-DI ------"
#METHOD="PGN-DI"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
# --n-iter $N_ITER --batch-size 40  --n-ensemble 5 \
#--input-transformation "DI(in_size=299, out_size=330)" --loss-function "cross_entropy" --grad-calculation "PGN_grad(zeta=3, delta=0.5, N=20)" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
#echo "------ PGN-admix ------"
#METHOD="PGN-admix"
#$ATTACK --json-path "./json_files/I-FGSM.json" --save-dir "${PATH_ADV_BASE}/${METHOD}" --csv-export-path $PATH_CSV \
#--source-model-path $ENSEMBLE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON \
# --n-iter $N_ITER --batch-size 25  --n-ensemble 5 \
#--input-transformation "admix(strength=0.2, n_samples=3)" --loss-function "cross_entropy" --grad-calculation "PGN_grad(zeta=3, delta=0.5, N=20)" \
#--backpropagation "nonlinear" --update-dir-calculation "sgd"
#
### 不要的
#echo "------ I-FGSM ------"
#METHOD="I-FGSM"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ random_start ------"
#METHOD="random_start"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ MI-FGSM ------"
#METHOD="MI-FGSM"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ NI ------"
#METHOD="NI"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ PI ------"
#METHOD="PI"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ VT ------"
#METHOD="VT"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ RAP ------"
#METHOD="RAP"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ DI2-FGSM ------"
#METHOD="DI2-FGSM"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ SI ------"
#METHOD="SI"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ admix ------"
#METHOD="admix"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ TI ------"
#METHOD="TI"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ MI-DI ------"
#METHOD="MI-DI"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ MI-DI-TI ------"
#METHOD="MI-DI-TI"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ MI-DI-TI-SI ------"
#METHOD="MI-DI-TI-SI"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ VMI ------"
#METHOD="VMI"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ VNI ------"
#METHOD="VNI"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ admix-RAP ------"
#METHOD="admix-RAP"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 400 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV
#
#echo "------ SI-RAP ------"
#METHOD="SI-RAP"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter ${N_ITER} --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ RD ------"
#METHOD="RD"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#--rfmodel-dir "NIPS2017/RD/resnet50/0.005" --source-model-refinement "sample_from_isotropic(std=0.005, n_models=10)"
#
#echo "------ GhostNet ------"
#METHOD="GhostNet"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE}
#
#echo "------ LinBP ------"
#METHOD="LinBP"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 300 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#--loss-function "linbp(linbp_layer='3_1')" --batch-size 100
#
#echo "------ SGM ------"
#METHOD="SGM"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#--grad-calculation "skip_gradient(gamma=0.2)" --batch-size ${BATCH_SIZE}
#
#echo "------ ILA ------"
#METHOD_BSL="ILA_BSL"
#BSL_ADV="${PATH_ADV_BASE}/$SOURCE/bsl_adv"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD_BSL}.json" --n-iter 10 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD_BSL}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#--bsl-adv-img-path $BSL_ADV --batch-size ${BATCH_SIZE}
#METHOD="ILA"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#--bsl-adv-img-path $BSL_ADV --batch-size 100 --loss-function "ila_loss(ila_layer='2')"
#
#echo "------ FIA ------"
#METHOD="FIA"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#--loss-function "fia_loss(fia_layer='2')" --batch-size 100
#
#echo "------ NAA ------"
#METHOD="NAA"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#--loss-function "naa_loss(naa_layer='2')" --batch-size 100
#
#echo "------ DRA ------"
#METHOD="DRA"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path "NIPS2017/DRA/DRA_resnet50.pth" --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#--batch-size ${BATCH_SIZE}
#
#echo "------ IAA ------"
#METHOD="IAA"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path "NIPS2017/IAA/resnet50" --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV \
#--batch-size ${BATCH_SIZE}
#
#echo "------ LGV ------"
#METHOD="LGV"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
#$ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#--source-model-refinement "stochastic_weight_collecting(collect=False)"
#
#echo "------ SWA ------"
#METHOD="SWA"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#--rfmodel-dir "NIPS2017/SWA/$SOURCE/swa_model" --source-model-refinement "stochastic_weight_averaging(collect=False)"
#
#echo "------ Bayesian_attack ------"
#METHOD="Bayesian_attack"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH $ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#--rfmodel-dir "NIPS2017/Bayesian_Attack/$SOURCE/swag_samples" \
#--source-model-refinement "sample_from_swag(collect=False, beta=0, scale=1.5, n_models=50)"
#
#echo "------ LGV-GhostNet ------"
#METHOD="LGV-GhostNet"
#$ATTACK --json-path "${PATH_JSON_BASE}/${METHOD}.json" --n-iter 100 --save-dir "${PATH_ADV_BASE}/$SOURCE/${METHOD}" \
#--source-model-path $SOURCE_PATH --target-model-path $TARGET_PATH \
#$ARGS_COMMON --csv-export-path $PATH_CSV --batch-size ${BATCH_SIZE} \
#--source-model-refinement "stochastic_weight_collecting(collect=False)"

