#!/bin/bash
# Step 5 (VARIANT=v1|v2): train + convert + test, with the notebook's exact model_train_eval arguments.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
export LD_LIBRARY_PATH=$(ls -d $WW/venvs/mww/lib/python3.10/site-packages/nvidia/*/lib | tr '\n' ':')
V=${VARIANT:-v1}; TD=$WW/train; [ $V != v1 ] && TD=$WW/train_$V
mkdir -p $TD; cd $TD
$WW/venvs/mww/bin/python -I $WW/scripts/04_config.py $V
$WW/venvs/mww/bin/python -I $WW/scripts/train_launcher.py \
--training_config="$TD/training_parameters.yaml" \
--train ${TRAIN:-1} \
--restore_checkpoint 1 \
--test_tf_nonstreaming 0 \
--test_tflite_nonstreaming 0 \
--test_tflite_nonstreaming_quantized 0 \
--test_tflite_streaming 0 \
--test_tflite_streaming_quantized 1 \
--use_weights "best_weights" \
mixednet \
--pointwise_filters "64,64,64,64" \
--repeat_in_block  "1, 1, 1, 1" \
--mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \
--residual_connection "0,0,0,0" \
--first_conv_filters 32 \
--first_conv_kernel_size 5 \
--stride 3
