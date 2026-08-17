#!/bin/bash

python evaluate.py \
    --model_path '/vols/dune/jmm224/t2knova/reweighting/saved_models/6D_5DMode_v2_0to4_cat_even_2/6D/XGB_model_6D.pkl' \
    --original_test_nD '/vols/dune/jmm224/t2knova/reweighting/saved_samples/6D_5DMode_v2_1/6D/original_test.csv' \
    --original_test '/vols/dune/jmm224/t2knova/reweighting/saved_samples/6D_5DMode_v2_1/21D/original_test.csv' \
    --target_test_nD '/vols/dune/jmm224/t2knova/reweighting/saved_samples/6D_5DMode_v2_1/6D/target_test.csv' \
    --target_test '/vols/dune/jmm224/t2knova/reweighting/saved_samples/6D_5DMode_v2_1/21D/target_test.csv' \
    --swd_distribution_target_3D '/vols/dune/jmm224/t2knova/reweighting/saved_swd_distribution/6D_5DMode_v2_0to5/3D/swd_distribution_3D.npy' \
    --swd_distribution_target_8D '/vols/dune/jmm224/t2knova/reweighting/saved_swd_distribution/6D_5DMode_v2_0to5/8D/swd_distribution_8D.npy' \
    --swd_distribution_target_21D '/vols/dune/jmm224/t2knova/reweighting/saved_swd_distribution/6D_5DModev2_modev2_1/21D/swd_distribution_21D.npy' \
    --swd_distribution_target_nD '/vols/dune/jmm224/t2knova/reweighting/saved_swd_distribution/6D_5DModev2_modev2_1/6D/swd_distribution_6D.npy' \
    --binning_file '/vols/dune/jmm224/t2knova/reweighting/binnings.json' \
    --output_path '/vols/dune/jmm224/t2knova/reweighting/Saved_evaluation/trained_modesv2_0to4_cat_even2_eval_modev2_1/' \
    --param_trained "Enu_true" "ELep" "CosLep" "W" "Eav" "Mode_v2"
