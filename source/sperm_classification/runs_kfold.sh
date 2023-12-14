#!/bin/bash
index_array=('seg_relu' 'ELU' 'LeakyReLU' 'relu')
# shellcheck disable=SC2068
for activation in ${index_array[@]}
do
  python tools/kfold.py \
  -ep 10 -bsize 64 -verbose 1 \
  -test "path_data_test.data" \
  -train "path_data_train.data" \
  -name "case_number-model_name" \
  -nk 10 -ck 1 \
  --activation_block "$activation" \
  -v "$activation-version-1.0" \
  --mode_model "hsc-v1" \
  --k_fold_path "./runs/k-fold" \
  --result_path "./runs/results" \
  -mode_labels "category"
done