#!/bin/bash
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

set -e -v
cd $(dirname $0)



#for converted quanti model inference
quanti_model_file="./model_output/yolov5s_672x672_nv12_quantized_model.onnx"
quanti_input_layout="NHWC"

#for original float model inference
original_model_file="./model_output/yolov5s_672x672_nv12_original_float_model.onnx"
original_input_layout="NCHW"

if [[ $1 =~ "origin" ]];  then
  model=$original_model_file
  layout=$original_input_layout
else
  model=$quanti_model_file
  layout=$quanti_input_layout
fi

# infer_image="../../../01_common/test_data/det_images/kite.jpg"

infer_image="/data/ai/yolov5_x3_v2/runs/exp23/test_batch0_pred.jpg"

# -----------------------------------------------------------------------------------------------------
# shell command "sh 04_inference.sh" runs quanti inference by default 
# If quanti model infer is intended, please run the shell via command "sh 04_inference.sh quanti"
# If float  model infer is intended, please run the shell via command "sh 04_inference.sh origin"
# -----------------------------------------------------------------------------------------------------

python3 -u ../../det_inference.py \
        --model ${model} \
        --image ${infer_image} \
        --input_layout ${layout}
