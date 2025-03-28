# Yolov5s

## Prepare model and data
1. YOLOv5s model
  1.1 corresponding pt file can be downloaded from URL: https://github.com/ultralytics/yolov5/releases/tag/v2.0
  1.2 verification md5sum: 
  2e296b5e31bf1e1b6b8ea4bf36153ea5  yolov5l.pt
  16150e35f707a2f07e7528b89c032308  yolov5m.pt
  42c681cf466c549ff5ecfe86bcc491a0  yolov5s.pt
  069a6baa2a741dec8a2d44a9083b6d6e  yolov5x.pt
  1.3 after download, convert pt file into ONNX file using the `https://github.com/ultralytics/yolov5/blob/v2.0/models/export.py` script
  1.4 As for YOLOv5 model, in terms of model structure, we modified some output nodes. As currently OpenExplorer XJ3 Toolchain doesn't support 5-dimensional Reshape, we deleted it in the prototxt and moved it to post-process. Meanwhile, we've aslo added a transpose operator to enable the node to dump NHWC. This is because in Horizon's ASIC, BPU hardware runs NHWC layout, therefore, BPU can directly dump results after some modifications, rather than introducing additional transpose in quantized model. For more details please refer to the text and table in the Benchmark section.
2. COCO verification dataset is used for computing model accuracy and can be downloaded from COCO official website: http://cocodataset.org/
3. Calibration dataset: extract 50 images from COCO verification dataset to serve as calibration dataset
4. origin float model accuracy : `[IoU=0.50:0.95] 0.352 [IoU=0.50] 0.542`

## how to use

使用的是这个 docker 

registry.cn-hangzhou.aliyuncs.com/l2493_fc/ai_toolchain_ubuntu_20_xj3_gpu:v2.6.6-py38

首先确保 docker 连接到宿主机的训练结果上，如我连接到 /data/ 下，对应的工作区如下，通过 docker 拿到 hb_mapper 命令跟量化仿真环境。

root@b23f439f427a:/data/ai/x3/horizon_xj3_open_explorer_v2.6.6/ddk/samples/ai_toolchain/horizon_model_convert_sample/04_detection/03_yolov5s/mapper#

### 01_check.sh

这里检查模型，和更换模型，默认 672 测试

```

model_type="onnx"
# onnx_model="../../../01_common/model_zoo/mapper/detection/yolov5_onnx_optimized/YOLOv5s.onnx"
onnx_model=/data/ai/yolov5_x3_v2/runs/exp23/weights/best.onnx
# onnx_model=/data/ai/yolov5_x3_v2/weights/yolov5s.onnx

```

### 02_preprocess.sh

准备量化验证数据，直接从训练里拉出来，不改也可用。

```

rm ./calibration_data_rgb_f32/*

# python3 ../../../data_preprocess.py \
#   --src_dir ../../../01_common/calibration_data/coco \
#   --dst_dir ./calibration_data_rgb_f32 \
#   --pic_ext .rgb \
#   --read_mode opencv \
#   --saved_data_type float32

python3 ../../../data_preprocess.py \
  --src_dir /data//ai/v5v2_data/data/valid/images \
  --dst_dir ./calibration_data_rgb_f32 \
  --pic_ext .rgb \
  --read_mode opencv \
  --saved_data_type float32

```

如果模型 672 改到 640 则要同步修改 preprocess.py 的图像大小

```
        PadResizeTransformer(target_size=(672, 672)),
        # PadResizeTransformer(target_size=(640, 640)),
```

### 03_build.sh

开始量化模型，注意修改 yolov5s_config.yaml 

```

config_file="./yolov5s_config.yaml"
model_type="onnx"
# build model
hb_mapper makertbin --config ${config_file}  \
                    --model-type  ${model_type}

```

yolov5s_config.yaml 

```

  # Onnx浮点网络数据模型文件
  # -----------------------------------------------------------
  # the model file of floating-point ONNX neural network data
  # onnx_model: '../../../01_common/model_zoo/mapper/detection/yolov5_onnx_optimized/YOLOv5s.onnx'
  # onnx_model: '/data/ai/yolov5_x3_v2/weights/yolov5s.onnx'
  onnx_model: '/data/ai/yolov5_x3_v2/runs/exp23/weights/best.onnx'

```

此时 model_output/yolov5s_672x672_nv12.bin 就是量化后板子可以使用的模型了。

### 04_inference.sh

验证数据无关正确，只是为了验证量化与浮点前后的精度比较。

此时用于对比仿真量化结果和浮点结果进行损失比较 如 origin 或 quanti 两个比较，并给出评估结果。

需要修改 postprocess.py 的输出通道大小

```

    # 640*640 2 class
    # model_output[0] = model_output[0].reshape([1, 80, 80, 3,
    #                                            7]).transpose([0, 3, 1, 2, 4])
    # model_output[1] = model_output[1].reshape([1, 40, 40, 3,
    #                                            7]).transpose([0, 3, 1, 2, 4])
    # model_output[2] = model_output[2].reshape([1, 20, 20, 3,
    #                                            7]).transpose([0, 3, 1, 2, 4])

    # 672*672 2 class 原有80分类的解析85=（80+4+1）修改为7=（2+4+1）
    model_output[0] = model_output[0].reshape([1, 84, 84, 3,
                                               7]).transpose([0, 3, 1, 2, 4])
    model_output[1] = model_output[1].reshape([1, 42, 42, 3,
                                               7]).transpose([0, 3, 1, 2, 4])
    model_output[2] = model_output[2].reshape([1, 21, 21, 3,
                                               7]).transpose([0, 3, 1, 2, 4])

    # model_output[0] = model_output[0].reshape([1, 84, 84, 3,
    #                                            85]).transpose([0, 3, 1, 2, 4])
    # model_output[1] = model_output[1].reshape([1, 42, 42, 3,
    #                                            85]).transpose([0, 3, 1, 2, 4])
    # model_output[2] = model_output[2].reshape([1, 21, 21, 3,
    #                                            85]).transpose([0, 3, 1, 2, 4])

```

还有输入大小和类别数量

```
    # yolov5s_config.NUM_CLASSES = 80
    # yolov5s_config.CLASSES = MSCOCODetMetric.class_names
    yolov5s_config.NUM_CLASSES = 2
    yolov5s_config.CLASSES = ["fire", "smoke"]# MSCOCODetMetric.class_names
    yolov5s_config.INPUT_SHAPE = (672, 672)
    # yolov5s_config.INPUT_SHAPE = (640, 640)
```

以及标签 dls_coco_classes.names ，这个只影响绘图显示的字符，不影响结果。

### 05_evaluate.sh

给模型来一场大考，用于优化数据集和清洗错误数据，挑出来标注后加进去训练。

