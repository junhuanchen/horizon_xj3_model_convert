#!/usr/bin/env sh
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

set -e -v
cd $(dirname $0) || exit

model_type="onnx"
# onnx_model="../../../01_common/model_zoo/mapper/detection/yolov5_onnx_optimized/YOLOv5s.onnx"
onnx_model=/data/ai/yolov5_x3_v2/runs/exp23/weights/best.onnx
# onnx_model=/data/ai/yolov5_x3_v2/weights/yolov5s.onnx

march="bernoulli2"

hb_mapper checker --model-type ${model_type} \
                  --model ${onnx_model} \
                  --march ${march}
