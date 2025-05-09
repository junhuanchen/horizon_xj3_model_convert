# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import numpy as np
import logging
import cv2
import re
import time
import colorsys
from easydict import EasyDict
from PIL import Image
from matplotlib import pyplot as plt
from multiprocessing.pool import ApplyResult
from coco_metric import MSCOCODetMetric
from horizon_tc_ui.utils import tool_utils


def get_yolov5s_config():
    yolov5s_config = EasyDict()
    yolov5s_config.ANCHORS = np.array([
        10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198,
        373, 326
    ]).reshape((3, 3, 2))
    yolov5s_config.STRIDES = np.array([8, 16, 32])
    # yolov5s_config.NUM_CLASSES = 80
    # yolov5s_config.CLASSES = MSCOCODetMetric.class_names
    yolov5s_config.NUM_CLASSES = 2
    yolov5s_config.CLASSES = ["fire", "smoke"]# MSCOCODetMetric.class_names
    yolov5s_config.INPUT_SHAPE = (672, 672)
    # yolov5s_config.INPUT_SHAPE = (640, 640)
    return yolov5s_config


def postprocess(model_output,
                model_hw_shape,
                origin_image=None,
                origin_img_shape=None,
                score_threshold=0.4,
                nms_threshold=0.45,
                dump_image=True):
    yolov5s_config = get_yolov5s_config()
    classes = yolov5s_config.CLASSES
    num_classes = yolov5s_config.NUM_CLASSES
    anchors = yolov5s_config.ANCHORS
    num_anchors = anchors.shape[0]
    strides = yolov5s_config.STRIDES
    input_shape = yolov5s_config.INPUT_SHAPE

    if origin_image is not None:
        org_height, org_width = origin_image.shape[1:3]
    else:
        org_height, org_width = origin_img_shape
    process_height, process_width = model_hw_shape

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

    pred_sbbox, pred_mbbox, pred_lbbox = model_output[0], model_output[
        1], model_output[2]

    pred_sbbox = yolov5s_decoder(pred_sbbox, num_anchors, num_classes,
                                 anchors[0], strides[0])
    pred_mbbox = yolov5s_decoder(pred_mbbox, num_anchors, num_classes,
                                 anchors[1], strides[1])
    pred_lbbox = yolov5s_decoder(pred_lbbox, num_anchors, num_classes,
                                 anchors[2], strides[2])
    pred_bbox = np.concatenate([
        np.reshape(pred_sbbox, (-1, 5 + num_classes)),
        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
        np.reshape(pred_lbbox, (-1, 5 + num_classes))
    ],
                               axis=0)

    bboxes = postprocess_boxes(pred_bbox, (org_height, org_width),
                               input_shape=(process_height, process_width),
                               score_threshold=score_threshold)
    nms_bboxes = nms(bboxes, nms_threshold)
    if dump_image and origin_image is not None:
        logging.info(f"detected item num: {len(nms_bboxes)}")
        draw_bboxs(origin_image[0], nms_bboxes)
    return nms_bboxes


def eval_postprocess(model_output,
                     model_hw_shape,
                     entry_dict,
                     score_threshold=0.001,
                     nms_threshold=0.65):

    bboxes_pr = postprocess(model_output,
                            model_hw_shape,
                            origin_img_shape=entry_dict[0]['origin_shape'],
                            score_threshold=score_threshold,
                            nms_threshold=nms_threshold,
                            dump_image=False)
    pred_result = []
    for one_bbox in bboxes_pr:
        one_result = {'bbox': one_bbox, 'mask': False}
        pred_result.append(one_result)
    return pred_result, entry_dict[0]['image_name']


def calc_accuracy(annotation_path, accuracy):
    metric = MSCOCODetMetric(annotation_path, with_mask=False)
    if isinstance(accuracy[0], ApplyResult):
        batch_result = [i.get() for i in accuracy]
    else:
        batch_result = list(accuracy)
    total_samples = 0
    with open("yolov5s_eval.log", "w") as eval_log_handle:
        logging.info("start to calc eval info...")
        for pred_result, filename in batch_result:
            metric.update(pred_result, filename)
            pred_result.sort(key=lambda x: x['bbox'][4], reverse=True)
            eval_log_handle.write(f"input_image_name: {filename} ")
            for one_result in pred_result:
                det_item = one_result['bbox']
                eval_log_handle.write(
                    f"{det_item[0]:6f},{det_item[1]:6f},{det_item[2]:6f},{det_item[3]:6f},{det_item[4]:6f},{det_item[5]:0.0f} "
                )
            eval_log_handle.write("\n")
    return metric.get()


def gen_report(metric_result):
    names, values = metric_result
    summary = values[0]
    summary = summary.splitlines()
    pattern = re.compile(r'(IoU.*?) .* (.*)$')
    tool_utils.report_flag_start('MAPPER-EVAL')
    for v in summary[0:2]:
        valid_data = pattern.findall(v)[0]
        logging.info("[%s] = %s" % (valid_data[0], valid_data[1]))
    tool_utils.report_flag_end('MAPPER-EVAL')


def get_classes(class_file_name='coco_classes.names'):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def draw_bboxs(image, bboxes, gt_classes_index=None, classes=get_classes()):
    """draw the bboxes in the original image
    """
    num_classes = len(classes)
    image_h, image_w, channel = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))

    fontScale = 0.5
    bbox_thick = int(0.6 * (image_h + image_w) / 600)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)

        if gt_classes_index == None:
            class_index = int(bbox[5])
            score = bbox[4]
        else:
            class_index = gt_classes_index[i]
            score = 1

        bbox_color = colors[class_index]
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        classes_name = classes[class_index]
        bbox_mess = '%s: %.2f' % (classes_name, score)
        t_size = cv2.getTextSize(bbox_mess,
                                 0,
                                 fontScale,
                                 thickness=bbox_thick // 2)[0]
        cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                      bbox_color, -1)
        cv2.putText(image,
                    bbox_mess, (c1[0], c1[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0),
                    bbox_thick // 2,
                    lineType=cv2.LINE_AA)
        logging.info("{} is in the picture with confidence:{:.4f}".format(
            classes_name, score))
        fig_name = f'demo.jpg'
        cv2.imwrite(fig_name, image)
    return image


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    calculate the nms for bboxes
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou), ), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou**2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def bboxes_iou(boxes1, boxes2):
    """calculate iou for a list of bboxes"""
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
                  (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
                  (boxes2[..., 3] - boxes2[..., 1])
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def postprocess_boxes(pred_bbox,
                      org_img_shape,
                      input_shape,
                      score_threshold=0.5):
    """post process boxes"""
    valid_scale = [0, np.inf]
    org_h, org_w = org_img_shape
    input_h, input_w = input_shape
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, :4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([
        pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
        pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5
    ],
                               axis=-1)

    # (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    resize_ratio = min(input_h / org_h, input_w / org_w)
    dw = (input_w - resize_ratio * org_w) / 2
    dh = (input_h - resize_ratio * org_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # clip the range of bbox
    pred_coor = np.concatenate([
        np.maximum(pred_coor[:, :2], [0, 0]),
        np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
    ],
                               axis=-1)
    # drop illegal boxes whose max < min
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                 (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # discard invalid boxes
    bboxes_scale = np.sqrt(
        np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale),
                                (bboxes_scale < valid_scale[1]))

    # discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate(
        [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def yolov5s_decoder(conv_output, num_anchors, num_classes, anchors, stride):
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Five dimension output: [batch_size, num_anchors, output_size, output_size, 5 + num_classes]
    batch_size = conv_output.shape[0]
    output_size = conv_output.shape[-2]
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    y = np.tile(
        np.arange(output_size, dtype=np.int32)[:, np.newaxis],
        [1, output_size])
    x = np.tile(
        np.arange(output_size, dtype=np.int32)[np.newaxis, :],
        [output_size, 1])
    xy_grid = np.concatenate([x[:, :, np.newaxis], y[:, :, np.newaxis]],
                             axis=-1)
    xy_grid = np.tile(xy_grid[np.newaxis, np.newaxis, :, :, :],
                      [batch_size, num_anchors, 1, 1, 1])
    xy_grid = xy_grid.astype(np.float32)

    pred_xy = (sigmoid(conv_raw_dxdy) * 2.0 - 0.5 + xy_grid) * stride
    pred_wh = (sigmoid(conv_raw_dwdh) *
               2.0)**2 * anchors[np.newaxis, :, np.newaxis, np.newaxis, :]
    pred_xywh = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_conf = sigmoid(conv_raw_conf)
    pred_prob = sigmoid(conv_raw_prob)

    decode_output = np.concatenate([pred_xywh, pred_conf, pred_prob], axis=-1)
    return decode_output
